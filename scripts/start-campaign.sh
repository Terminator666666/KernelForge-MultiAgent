#!/usr/bin/env bash
# Start a closed-loop optimization campaign

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Usage
if [ $# -lt 2 ]; then
    echo "Usage: $0 <family> <max_rounds> [target_speedup]"
    echo ""
    echo "Arguments:"
    echo "  family         - 主线算子 family（dsa_sparse_attention, gdn_prefill, dsa_topk_indexer）"
    echo "  max_rounds     - Maximum number of optimization rounds (e.g., 10)"
    echo "  target_speedup - Optional target speedup (e.g., 2.0 for 2x)"
    echo ""
    echo "Example:"
    echo "  $0 dsa_sparse_attention 10 2.0"
    exit 1
fi

FAMILY="$1"
MAX_ROUNDS="$2"
TARGET_SPEEDUP="${3:-}"

CANONICAL_FAMILY="$("$PYTHON_BIN" "$PROJECT_ROOT/scripts/operator_policy.py" canonical "$FAMILY" 2>/dev/null || true)"
if [ -z "$CANONICAL_FAMILY" ]; then
    echo "Error: unsupported family: $FAMILY"
    echo "Only the three mainline families are allowed:"
    echo "  dsa_sparse_attention"
    echo "  gdn_prefill"
    echo "  dsa_topk_indexer"
    exit 1
fi
FAMILY="$CANONICAL_FAMILY"

echo "=========================================="
echo "KernelForge-MultiAgent Campaign"
echo "=========================================="
echo "Family:        $FAMILY"
echo "Max Rounds:    $MAX_ROUNDS"
echo "Target Speedup: ${TARGET_SPEEDUP:-auto}"
echo "=========================================="
echo ""

# Initialize reference archive if not exists
REFERENCE_DIR="$PROJECT_ROOT/reference/$FAMILY"
if [ ! -d "$REFERENCE_DIR" ]; then
    echo "📁 Initializing reference/$FAMILY/ archive..."
    mkdir -p "$REFERENCE_DIR/variants"
    
    # Create README.md
    cat > "$REFERENCE_DIR/README.md" << 'README_EOF'
# {{FAMILY}} Optimization Family

## Current Best Variant
- Variant ID: (not established)
- Speedup: 1.0x (baseline)
- Status: 🔴 Awaiting Round 0

## Optimization History
(No rounds completed)

## Variant Tree
(To be built)

## Exploration Directions
1. TBD after Round 0 baseline

---

**Last Updated**: $(date -Iseconds)
**Maintainer**: KernelForge-MultiAgent
README_EOF
    sed -i "s/{{FAMILY}}/$FAMILY/g" "$REFERENCE_DIR/README.md"
    
    # Create TRAPS.md
    cat > "$REFERENCE_DIR/TRAPS.md" << 'TRAPS_EOF'
# {{FAMILY}} Optimization Traps

## 🚨 Known Traps

(No traps recorded yet. Traps will be accumulated as the campaign progresses.)

---

## How to Use This File

### Before Optimizing
1. Read all known traps
2. Mark risk points in draft.md
3. Design prevention measures

### When Encountering Issues
1. Check if symptoms match known traps
2. Apply corresponding solutions
3. If new trap, record it here

### After Optimization
1. Review if any traps were triggered
2. Add newly discovered traps
3. Update prevention measures

---

**Last Updated**: $(date -Iseconds)
TRAPS_EOF
    sed -i "s/{{FAMILY}}/$FAMILY/g" "$REFERENCE_DIR/TRAPS.md"
    
    # Create baseline.json placeholder from the mainline family policy
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/operator_policy.py" template "$FAMILY" > "$REFERENCE_DIR/baseline.json"
    
    # Create solutions.jsonl
    touch "$REFERENCE_DIR/solutions.jsonl"
    
    echo "✅ Reference archive initialized"
    echo ""
fi

# Create rounds directory
ROUNDS_DIR="$PROJECT_ROOT/rounds"
mkdir -p "$ROUNDS_DIR"

# Start campaign
echo "🚀 Starting campaign..."
echo ""
echo "Master Agent will now orchestrate $MAX_ROUNDS rounds."
echo "Each round follows the 10-step loop:"
echo "  derive → brief → optimize → benchmark → validate"
echo "  → compare → decide → document → lessons → plan"
echo ""
echo "📝 Instructions:"
echo "1. Master Agent will create BRIEF.md for each round"
echo "2. Sub Agent (you) will optimize using Humanize RLCR"
echo "3. Evaluation will run automatically"
echo "4. Master Agent will make decisions"
echo ""
echo "To start Round 0, run:"
echo "  ./scripts/run-round.sh $FAMILY 0"
echo ""
echo "Or for interactive mode:"
echo "  cd rounds/round-0/$FAMILY"
echo "  # Master will guide you through the process"

# Save campaign config
CAMPAIGN_CONFIG="$ROUNDS_DIR/campaign-$FAMILY.json"
cat > "$CAMPAIGN_CONFIG" << CONFIG_EOF
{
  "family": "$FAMILY",
  "max_rounds": $MAX_ROUNDS,
  "target_speedup": ${TARGET_SPEEDUP:-null},
  "current_round": 0,
  "status": "initialized",
  "started_at": "$(date -Iseconds)"
}
CONFIG_EOF

echo ""
echo "✅ Campaign initialized"
echo "📄 Config: $CAMPAIGN_CONFIG"
echo ""
echo "=========================================="
echo "Ready to start Round 0!"
echo "=========================================="
