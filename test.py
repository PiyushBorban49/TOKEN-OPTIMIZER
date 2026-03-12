"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     TOKEN OPTIMIZER — 5-LAYER PIPELINE                     ║
║                                                                            ║
║  Layer 1: Ingestion         (Text cleanup, no LLM)                         ║
║  Layer 2: Logic-Anchor      (Regex/keyword mask, deterministic)            ║
║  Layer 3: Compressor        (SLM — Qwen2-1.5B via Ollama)                  ║
║  Layer 4: Quantizer         (Symbolic shorthand mapping)                   ║
║  Layer 5: Flagship LLM      (Final interpretation via OpenAI/Ollama)       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import re
import os
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import requests

# ── Fix Windows terminal encoding ────────────────────────────────────────────
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ─────────────────────────── Constants ────────────────────────────────────────

BASE_DIR = Path(__file__).parent
PROTECTED_OPS_PATH = BASE_DIR / "protected_operators.json"
QUANTIZATION_MAP_PATH = BASE_DIR / "quantization_map.json"
OLLAMA_URL = "http://localhost:11434/api/generate"
SLM_MODEL = "qwen2:1.5b"

# ─────────────────────────── Data Classes ─────────────────────────────────────

@dataclass
class LayerResult:
    """Output of each pipeline layer."""
    layer_name: str
    text: str
    token_count: int
    metadata: dict = field(default_factory=dict)
    processing_time: float = 0.0


@dataclass
class LogicMask:
    """Token classification from Layer 2."""
    protected_tokens: list[str]
    safe_to_prune: list[str]
    original_tokens: list[str]
    mask: list[bool]  # True = protected, False = safe to prune


# ─────────────────────────── Utilities ────────────────────────────────────────

class Colors:
    """ANSI color codes for terminal output."""
    HEADER  = "\033[95m"
    BLUE    = "\033[94m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RESET   = "\033[0m"
    MAGENTA = "\033[35m"
    WHITE   = "\033[97m"
    BG_DARK = "\033[48;5;235m"


def count_tokens(text: str) -> int:
    """Simple whitespace tokenizer for counting."""
    return len(text.split()) if text.strip() else 0


def print_banner():
    """Print the startup banner."""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
    ╔══════════════════════════════════════════════════════════════╗
    ║          ⚡ TOKEN OPTIMIZER — 5-LAYER PIPELINE ⚡           ║
    ║                                                              ║
    ║   L1  Ingestion          ➜  Text cleanup                    ║
    ║   L2  Logic-Anchor       ➜  Deterministic mask              ║
    ║   L3  SLM Compressor     ➜  Qwen2-1.5B via Ollama          ║
    ║   L4  Quantizer          ➜  Symbolic shorthand              ║
    ║   L5  Flagship LLM       ➜  Final interpretation            ║
    ╚══════════════════════════════════════════════════════════════╝
{Colors.RESET}"""
    print(banner)


def print_layer_result(result: LayerResult, original_tokens: int):
    """Pretty-print a single layer's output."""
    saved = original_tokens - result.token_count
    pct = (saved / original_tokens * 100) if original_tokens > 0 else 0

    # Layer header
    color_map = {
        "Layer 1 — Ingestion":     Colors.BLUE,
        "Layer 2 — Logic Anchor":  Colors.YELLOW,
        "Layer 3 — Compressor":    Colors.MAGENTA,
        "Layer 4 — Quantizer":     Colors.CYAN,
        "Layer 5 — Flagship LLM":  Colors.GREEN,
    }
    color = color_map.get(result.layer_name, Colors.WHITE)

    print(f"\n{color}{Colors.BOLD}{'━' * 66}")
    print(f"  📦  {result.layer_name}")
    print(f"{'━' * 66}{Colors.RESET}")
    print(f"  {Colors.DIM}Processing time: {result.processing_time:.3f}s{Colors.RESET}")

    # Output text
    print(f"\n  {Colors.BOLD}Output:{Colors.RESET}")
    print(f"  {Colors.WHITE}{result.text}{Colors.RESET}")

    # Token stats
    print(f"\n  {Colors.DIM}Tokens: {result.token_count}  │  "
          f"Saved: {saved} ({pct:.1f}%){Colors.RESET}")

    # Metadata
    if result.metadata:
        print(f"\n  {Colors.DIM}Metadata:{Colors.RESET}")
        for key, value in result.metadata.items():
            if isinstance(value, list):
                display = ", ".join(str(v) for v in value[:15])
                if len(value) > 15:
                    display += f"  … (+{len(value) - 15} more)"
                print(f"    {Colors.DIM}• {key}: [{display}]{Colors.RESET}")
            else:
                print(f"    {Colors.DIM}• {key}: {value}{Colors.RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYER 1 — INGESTION  (No LLM)
# ═══════════════════════════════════════════════════════════════════════════════

class Layer1_Ingestion:
    """
    Text cleanup and normalization.
    - Strips extra whitespace
    - Standardizes unicode symbols (°, ≥, ≤, →)
    - Normalizes quotes, dashes, ellipses
    - Converts written-out units to symbols
    """

    SYMBOL_MAP = {
        "degrees celsius":  "°C",
        "degrees fahrenheit": "°F",
        "degree celsius":   "°C",
        "degree fahrenheit": "°F",
        "deg c":            "°C",
        "deg f":            "°F",
        "greater than or equal to": ">=",
        "less than or equal to":    "<=",
        "greater than":     ">",
        "less than":        "<",
        "not equal to":     "!=",
        "equal to":         "==",
    }

    def process(self, raw_text: str) -> LayerResult:
        start = time.time()
        text = raw_text.strip()

        # Normalize unicode
        text = unicodedata.normalize("NFKC", text)

        # Standardize quotes and dashes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        text = re.sub(r"[–—]", "-", text)
        text = re.sub(r"…", "...", text)

        # Convert written-out units/symbols (case-insensitive)
        for phrase, symbol in self.SYMBOL_MAP.items():
            text = re.sub(
                re.escape(phrase),
                symbol,
                text,
                flags=re.IGNORECASE,
            )

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        elapsed = time.time() - start
        return LayerResult(
            layer_name="Layer 1 — Ingestion",
            text=text,
            token_count=count_tokens(text),
            metadata={"operations": ["unicode_norm", "symbol_standardize", "whitespace_collapse"]},
            processing_time=elapsed,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYER 2 — LOGIC-ANCHOR PARSER  (Deterministic / Regex)
# ═══════════════════════════════════════════════════════════════════════════════

class Layer2_LogicAnchorParser:
    """
    Deterministic safety net.
    Scans tokens against protected_operators.json and produces a LogicMask
    that marks every token as PROTECTED or SAFE-TO-PRUNE.
    """

    def __init__(self):
        self.protected_keywords: set[str] = set()
        self.protected_patterns: list[re.Pattern] = []
        self._load_operators()

    def _load_operators(self):
        """Load protected operators from JSON config."""
        if not PROTECTED_OPS_PATH.exists():
            raise FileNotFoundError(
                f"Missing config: {PROTECTED_OPS_PATH}\n"
                "Create 'protected_operators.json' in the project root."
            )

        with open(PROTECTED_OPS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        for category, keywords in data.items():
            for kw in keywords:
                self.protected_keywords.add(kw.lower())
                # Multi-word phrases become regex patterns
                if " " in kw:
                    self.protected_patterns.append(
                        re.compile(re.escape(kw), re.IGNORECASE)
                    )

        # Add comparison operator symbols
        self.protected_keywords.update({">", "<", ">=", "<=", "==", "!="})

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize preserving comparison operators and punctuation."""
        # Split on whitespace but keep comparison operators intact
        tokens = re.findall(r"[><=!]+|[\w°]+|[^\s\w]", text)
        return tokens

    def _is_protected(self, token: str) -> bool:
        """Check if a single token matches any protected keyword."""
        clean = token.strip(".,;:!?\"'()[]{}").lower()
        return clean in self.protected_keywords

    def _is_number(self, token: str) -> bool:
        """Numbers are always protected (thresholds, limits, etc)."""
        clean = token.strip(".,;:!?\"'()[]{}°CF")
        try:
            float(clean)
            return True
        except ValueError:
            return False

    def process(self, text: str) -> tuple[LayerResult, LogicMask]:
        start = time.time()
        tokens = self._tokenize(text)

        protected = []
        safe = []
        mask = []

        for token in tokens:
            if self._is_protected(token) or self._is_number(token):
                protected.append(token)
                mask.append(True)
            else:
                safe.append(token)
                mask.append(False)

        logic_mask = LogicMask(
            protected_tokens=protected,
            safe_to_prune=safe,
            original_tokens=tokens,
            mask=mask,
        )

        # Build annotated text showing the mask
        annotated_parts = []
        for token, is_prot in zip(tokens, mask):
            if is_prot:
                annotated_parts.append(f"[{token}]")
            else:
                annotated_parts.append(token)
        annotated_text = " ".join(annotated_parts)

        elapsed = time.time() - start
        result = LayerResult(
            layer_name="Layer 2 — Logic Anchor",
            text=annotated_text,
            token_count=count_tokens(text),
            metadata={
                "protected": protected,
                "safe_to_prune": safe,
                "protected_count": len(protected),
                "safe_count": len(safe),
            },
            processing_time=elapsed,
        )
        return result, logic_mask


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYER 3 — COMPRESSOR  (SLM — Qwen2-1.5B via Ollama)
# ═══════════════════════════════════════════════════════════════════════════════

class Layer3_Compressor:
    """
    Uses a Small Language Model (Qwen2-1.5B) to prune low-information tokens
    while strictly respecting the Logic Mask from Layer 2.

    Falls back to a rule-based stopword remover if Ollama is unreachable.
    """

    STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "about", "between", "out", "its", "it", "that",
        "this", "these", "those", "there", "here", "very", "really", "just",
        "quite", "rather", "somewhat", "usually", "often", "sometimes",
        "also", "too", "so", "then", "than", "such", "much", "many",
    }

    def _call_ollama(self, text: str, logic_mask: LogicMask) -> Optional[str]:
        """Send compression request to Ollama."""
        protected_str = ", ".join(set(logic_mask.protected_tokens))

        system_prompt = f"""You are a token compression engine. Your job is to compress the input text
into the shortest possible form while PRESERVING ALL LOGICAL MEANING.

CRITICAL RULES:
1. NEVER remove or alter these protected tokens: {protected_str}
2. Remove filler words (the, a, an, is, usually, etc.)
3. Keep subjects and objects that carry meaning (e.g., "fan", "system")
4. Replace verbose phrases with shorter alternatives
5. Preserve all numbers, operators, and conditions EXACTLY
6. Output ONLY the compressed text, nothing else
7. Do NOT add any explanation or commentary"""

        payload = {
            "model": SLM_MODEL,
            "prompt": f"Compress this text: \"{text}\"",
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 100,
            },
        }

        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=30)
            resp.raise_for_status()
            result = resp.json().get("response", "").strip()

            # Validate: ensure all protected tokens survived
            result_lower = result.lower()
            for token in logic_mask.protected_tokens:
                clean = token.strip(".,;:!?").lower()
                if clean and len(clean) > 1 and clean not in result_lower:
                    # Protected token was dropped — reject SLM output
                    print(f"  {Colors.RED}⚠ SLM dropped protected token '{token}' — "
                          f"falling back to heuristic{Colors.RESET}")
                    return None

            return result if result else None

        except (requests.ConnectionError, requests.Timeout):
            print(f"  {Colors.YELLOW}⚠ Ollama not reachable — using heuristic fallback{Colors.RESET}")
            return None
        except Exception as e:
            print(f"  {Colors.YELLOW}⚠ Ollama error: {e} — using heuristic fallback{Colors.RESET}")
            return None

    def _heuristic_compress(self, text: str, logic_mask: LogicMask) -> str:
        """Rule-based fallback: remove stopwords that aren't protected."""
        result_tokens = []
        for token, is_protected in zip(logic_mask.original_tokens, logic_mask.mask):
            if is_protected:
                result_tokens.append(token)
            elif token.strip(".,;:!?\"'").lower() not in self.STOPWORDS:
                result_tokens.append(token)
        return " ".join(result_tokens)

    def process(self, text: str, logic_mask: LogicMask) -> LayerResult:
        start = time.time()

        # Try SLM first, fall back to heuristic
        compressed = self._call_ollama(text, logic_mask)
        method = "slm_qwen2"

        if compressed is None:
            compressed = self._heuristic_compress(text, logic_mask)
            method = "heuristic_fallback"

        # Clean up the output
        compressed = re.sub(r"\s+", " ", compressed).strip()
        compressed = compressed.strip('"').strip("'")

        elapsed = time.time() - start
        return LayerResult(
            layer_name="Layer 3 — Compressor",
            text=compressed,
            token_count=count_tokens(compressed),
            metadata={"method": method, "model": SLM_MODEL if method == "slm_qwen2" else "N/A"},
            processing_time=elapsed,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYER 4 — QUANTIZER  (Hardcoded Mapping)
# ═══════════════════════════════════════════════════════════════════════════════

class Layer4_Quantizer:
    """
    Converts common phrases into symbolic shorthand
    using the quantization_map.json dictionary.
    """

    def __init__(self):
        self.phrase_map: dict[str, str] = {}
        self.unit_map: dict[str, str] = {}
        self._load_maps()

    def _load_maps(self):
        """Load quantization maps from JSON."""
        if not QUANTIZATION_MAP_PATH.exists():
            raise FileNotFoundError(
                f"Missing config: {QUANTIZATION_MAP_PATH}\n"
                "Create 'quantization_map.json' in the project root."
            )

        with open(QUANTIZATION_MAP_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.phrase_map = data.get("phrase_to_symbol", {})
        self.unit_map = data.get("unit_symbols", {})

    def process(self, text: str) -> LayerResult:
        start = time.time()
        quantized = text
        replacements = []

        # Sort phrases by length (longest first) to avoid partial matches
        sorted_phrases = sorted(self.phrase_map.items(), key=lambda x: len(x[0]), reverse=True)

        for phrase, symbol in sorted_phrases:
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            if pattern.search(quantized):
                quantized = pattern.sub(symbol, quantized)
                replacements.append(f"{phrase} → {symbol}")

        # Apply unit symbol replacements
        for unit, symbol in self.unit_map.items():
            if unit in quantized:
                quantized = quantized.replace(unit, symbol)
                replacements.append(f"{unit} → {symbol}")

        # Clean up: remove extra spaces around operators
        quantized = re.sub(r"\s*([><=!]+)\s*", r"\1", quantized)
        # Collapse whitespace
        quantized = re.sub(r"\s+", " ", quantized).strip()

        elapsed = time.time() - start
        return LayerResult(
            layer_name="Layer 4 — Quantizer",
            text=quantized,
            token_count=count_tokens(quantized),
            metadata={"replacements": replacements, "replacement_count": len(replacements)},
            processing_time=elapsed,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYER 5 — FLAGSHIP LLM  (Final Interpretation)
# ═══════════════════════════════════════════════════════════════════════════════

class Layer5_FlagshipLLM:
    """
    Sends the compressed prompt to a flagship LLM for final interpretation.
    Uses Ollama with a larger model if available, otherwise prints the result.
    """

    def _call_ollama_flagship(self, compressed_prompt: str, original_text: str) -> Optional[str]:
        """Use Ollama with a capable model for final interpretation."""
        system_prompt = """You are am Industrial AI safety system that interprets compressed operational rules.
You receive compressed, symbolic logic and must produce a clear, actionable safety warning or rule interpretation.

RULES:
1. Expand symbolic shorthand back to human-readable form
2. Identify: Condition, Exception, and Action
3. Produce a clear, concise safety warning
4. Output ONLY the interpreted rule, nothing else"""

        payload = {
            "model": SLM_MODEL,
            "prompt": (
                f"Interpret this compressed operational rule and produce a clear safety warning:\n"
                f"Compressed: {compressed_prompt}\n"
                f"Original context: {original_text}"
            ),
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 200,
            },
        }

        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as e:
            print(f"  {Colors.YELLOW}⚠ Flagship LLM error: {e}{Colors.RESET}")
            return None

    def process(self, compressed_text: str, original_text: str) -> LayerResult:
        start = time.time()

        interpretation = self._call_ollama_flagship(compressed_text, original_text)
        method = "ollama"

        if interpretation is None:
            # Fallback: simple rule extraction
            interpretation = self._rule_based_interpret(compressed_text)
            method = "rule_based_fallback"

        elapsed = time.time() - start
        return LayerResult(
            layer_name="Layer 5 — Flagship LLM",
            text=interpretation,
            token_count=count_tokens(interpretation),
            metadata={"method": method},
            processing_time=elapsed,
        )

    def _rule_based_interpret(self, compressed: str) -> str:
        """Simple rule-based interpretation fallback."""
        parts = {
            "condition": "",
            "exception": "",
            "action": "",
        }

        lower = compressed.lower()

        if "when" in lower or "if" in lower:
            # Extract condition
            for kw in ["when", "if"]:
                if kw in lower:
                    idx = lower.index(kw)
                    rest = compressed[idx:]
                    # Find the next logic keyword
                    for stop in ["unless", "but", "except"]:
                        if stop in rest.lower():
                            stop_idx = rest.lower().index(stop)
                            parts["condition"] = rest[:stop_idx].strip()
                            parts["exception"] = rest[stop_idx:].strip()
                            break
                    else:
                        parts["condition"] = rest.strip()
                    break

        if "fails" in lower:
            parts["action"] = "SYSTEM FAILURE"
        elif "triggers" in lower:
            parts["action"] = "TRIGGER ACTIVATED"
        elif "activates" in lower:
            parts["action"] = "SYSTEM ACTIVATED"
        else:
            parts["action"] = "ACTION REQUIRED"

        result_lines = []
        result_lines.append(f"⚠ Safety Rule Interpretation (fallback):")
        if parts["condition"]:
            result_lines.append(f"  Condition: {parts['condition']}")
        if parts["exception"]:
            result_lines.append(f"  Exception: {parts['exception']}")
        result_lines.append(f"  Action: {parts['action']}")

        return "\n".join(result_lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class TokenOptimizerPipeline:
    """Orchestrates the 5-layer token optimization pipeline."""

    def __init__(self):
        self.layer1 = Layer1_Ingestion()
        self.layer2 = Layer2_LogicAnchorParser()
        self.layer3 = Layer3_Compressor()
        self.layer4 = Layer4_Quantizer()
        self.layer5 = Layer5_FlagshipLLM()

    def run(self, raw_text: str) -> dict:
        """Execute the full pipeline and return results for all layers."""

        print(f"\n{Colors.BOLD}{Colors.WHITE}{'═' * 66}")
        print(f"  📝 INPUT")
        print(f"{'═' * 66}{Colors.RESET}")
        print(f"  {raw_text}")
        original_tokens = count_tokens(raw_text)
        print(f"  {Colors.DIM}Tokens: {original_tokens}{Colors.RESET}")

        results = {}
        total_start = time.time()

        # ── Layer 1: Ingestion ──
        r1 = self.layer1.process(raw_text)
        print_layer_result(r1, original_tokens)
        results["layer1"] = r1

        # ── Layer 2: Logic Anchor ──
        r2, logic_mask = self.layer2.process(r1.text)
        print_layer_result(r2, original_tokens)
        results["layer2"] = r2

        # ── Layer 3: Compressor ──
        r3 = self.layer3.process(r1.text, logic_mask)
        print_layer_result(r3, original_tokens)
        results["layer3"] = r3

        # ── Layer 4: Quantizer ──
        r4 = self.layer4.process(r3.text)
        print_layer_result(r4, original_tokens)
        results["layer4"] = r4

        # ── Layer 5: Flagship LLM ──
        r5 = self.layer5.process(r4.text, raw_text)
        print_layer_result(r5, original_tokens)
        results["layer5"] = r5

        # ── Summary ──
        total_time = time.time() - total_start
        final_tokens = r4.token_count
        saved = original_tokens - final_tokens
        pct = (saved / original_tokens * 100) if original_tokens > 0 else 0

        print(f"\n{Colors.GREEN}{Colors.BOLD}{'═' * 66}")
        print(f"  📊 PIPELINE SUMMARY")
        print(f"{'═' * 66}{Colors.RESET}")
        print(f"  {Colors.WHITE}Original tokens:    {original_tokens}{Colors.RESET}")
        print(f"  {Colors.WHITE}Final tokens (L4):  {final_tokens}{Colors.RESET}")
        print(f"  {Colors.GREEN}{Colors.BOLD}Tokens saved:       {saved}  ({pct:.1f}%){Colors.RESET}")
        print(f"  {Colors.DIM}Total time:         {total_time:.3f}s{Colors.RESET}")
        print(f"  {Colors.DIM}Compression ratio:  {original_tokens}:{final_tokens}{Colors.RESET}")

        # Show the compression journey
        print(f"\n  {Colors.CYAN}{Colors.BOLD}Compression Journey:{Colors.RESET}")
        print(f"  {Colors.DIM}L1: {r1.text}{Colors.RESET}")
        print(f"  {Colors.DIM}L3: {r3.text}{Colors.RESET}")
        print(f"  {Colors.CYAN}{Colors.BOLD}L4: {r4.text}{Colors.RESET}")
        print(f"{Colors.GREEN}{'═' * 66}{Colors.RESET}\n")

        return results


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN — Interactive REPL
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Interactive REPL for the Token Optimizer."""
    print_banner()

    # Check Ollama connectivity
    print(f"  {Colors.DIM}Checking Ollama connectivity...{Colors.RESET}", end=" ")
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            if any(SLM_MODEL in m for m in models):
                print(f"{Colors.GREEN}✓ Connected — {SLM_MODEL} available{Colors.RESET}")
            else:
                print(f"{Colors.YELLOW}⚠ Connected but {SLM_MODEL} not found. "
                      f"Available: {', '.join(models) or 'none'}{Colors.RESET}")
                print(f"  {Colors.DIM}Run: ollama pull {SLM_MODEL}{Colors.RESET}")
        else:
            print(f"{Colors.YELLOW}⚠ Ollama responded with status {resp.status_code}{Colors.RESET}")
    except Exception:
        print(f"{Colors.YELLOW}⚠ Ollama not running — using heuristic fallback for Layer 3{Colors.RESET}")
        print(f"  {Colors.DIM}Start Ollama: https://ollama.com{Colors.RESET}")

    pipeline = TokenOptimizerPipeline()

    # Example prompt for quick demo
    example = "The system usually fails when temperature exceeds 40°C, unless the backup fan is active."

    print(f"\n{Colors.BOLD}  Enter a business rule to optimize (or press Enter for demo):{Colors.RESET}")
    print(f"  {Colors.DIM}Type 'quit' or 'exit' to stop.{Colors.RESET}\n")

    while True:
        try:
            user_input = input(f"  {Colors.CYAN}▶{Colors.RESET}  ").strip()

            if user_input.lower() in ("quit", "exit", "q"):
                print(f"\n  {Colors.DIM}Goodbye! 👋{Colors.RESET}\n")
                break

            if not user_input:
                user_input = example
                print(f"  {Colors.DIM}Using demo: {example}{Colors.RESET}")

            pipeline.run(user_input)

        except KeyboardInterrupt:
            print(f"\n\n  {Colors.DIM}Interrupted. Goodbye! 👋{Colors.RESET}\n")
            break
        except Exception as e:
            print(f"\n  {Colors.RED}Error: {e}{Colors.RESET}\n")


if __name__ == "__main__":
    main()
