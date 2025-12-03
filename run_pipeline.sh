#!/bin/bash

# Teacher-Guided GRPO Pipeline Runner
# This script runs the complete pipeline with error handling and logging

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs"}
LOG_DIR=${LOG_DIR:-"./logs"}
TEACHER_API=${TEACHER_API:-"http://localhost:8129/v1/chat/completions"}

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Teacher-Guided GRPO Pipeline"
echo "=========================================="
echo "Output Directory: $OUTPUT_DIR"
echo "Log Directory: $LOG_DIR"
echo "Teacher API: $TEACHER_API"
echo "=========================================="

# Function to log with timestamp
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

# Check if vLLM server is running
check_teacher() {
    log "Checking teacher model connection..."
    if python synthesis/teacher_query.py > "$LOG_DIR/teacher_test.log" 2>&1; then
        log "✓ Teacher model connection successful"
        return 0
    else
        error "✗ Teacher model connection failed"
        error "Please ensure vLLM server is running at $TEACHER_API"
        cat "$LOG_DIR/teacher_test.log"
        return 1
    fi
}

# Stage 1: Data Preparation
run_data_stage() {
    log "Starting Stage 1: Data Preparation"
    if python main.py --stage data --output_dir "$OUTPUT_DIR" > "$LOG_DIR/stage1_data.log" 2>&1; then
        log "✓ Stage 1 complete"
        return 0
    else
        error "✗ Stage 1 failed"
        tail -n 50 "$LOG_DIR/stage1_data.log"
        return 1
    fi
}

# Stage 2: Teacher Synthesis
run_synthesis_stage() {
    log "Starting Stage 2: Teacher Synthesis"
    if python main.py --stage synthesis --output_dir "$OUTPUT_DIR" > "$LOG_DIR/stage2_synthesis.log" 2>&1; then
        log "✓ Stage 2 complete"
        return 0
    else
        error "✗ Stage 2 failed"
        tail -n 50 "$LOG_DIR/stage2_synthesis.log"
        return 1
    fi
}

# Stage 3: Training
run_training_stage() {
    log "Starting Stage 3: GRPO Training"
    if python main.py --stage train --output_dir "$OUTPUT_DIR" 2>&1 | tee "$LOG_DIR/stage3_training.log"; then
        log "✓ Stage 3 complete"
        return 0
    else
        error "✗ Stage 3 failed"
        tail -n 50 "$LOG_DIR/stage3_training.log"
        return 1
    fi
}

# Stage 4: Evaluation
run_evaluation_stage() {
    log "Starting Stage 4: Evaluation"
    if python main.py --stage eval --output_dir "$OUTPUT_DIR" 2>&1 | tee "$LOG_DIR/stage4_evaluation.log"; then
        log "✓ Stage 4 complete"
        return 0
    else
        error "✗ Stage 4 failed"
        tail -n 50 "$LOG_DIR/stage4_evaluation.log"
        return 1
    fi
}

# Main execution
main() {
    local start_time=$(date +%s)

    # Check teacher connection
    if ! check_teacher; then
        exit 1
    fi

    # Run stages
    if ! run_data_stage; then
        error "Pipeline failed at Stage 1"
        exit 1
    fi

    if ! run_synthesis_stage; then
        error "Pipeline failed at Stage 2"
        exit 1
    fi

    if ! run_training_stage; then
        error "Pipeline failed at Stage 3"
        exit 1
    fi

    if ! run_evaluation_stage; then
        error "Pipeline failed at Stage 4"
        exit 1
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))

    echo ""
    log "=========================================="
    log "Pipeline Complete!"
    log "=========================================="
    log "Total time: ${hours}h ${minutes}m ${seconds}s"
    log "Model saved to: $OUTPUT_DIR/final_model"
    log "Results saved to: $OUTPUT_DIR/evaluation_results.json"
    log "Logs saved to: $LOG_DIR"
    log "=========================================="
}

# Parse arguments
case "${1:-all}" in
    all)
        main
        ;;
    data)
        run_data_stage
        ;;
    synthesis)
        check_teacher && run_synthesis_stage
        ;;
    train)
        run_training_stage
        ;;
    eval)
        run_evaluation_stage
        ;;
    test-teacher)
        check_teacher
        ;;
    *)
        echo "Usage: $0 {all|data|synthesis|train|eval|test-teacher}"
        exit 1
        ;;
esac
