#!/bin/bash
################################################################################
# Download Progress Monitor
#
# Run this in a separate terminal to watch download progress in real-time
#
# Usage:
#   bash monitor_download.sh [data_root]
################################################################################

DATA_ROOT="${1:-./data}"
KITTI_RAW="${DATA_ROOT}/kitti_raw"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Expected totals (for 20 drives)
EXPECTED_DRIVES=20
AVG_DRIVE_SIZE_MB=1500  # Average ~1.5GB per drive
TOTAL_SIZE_MB=$((EXPECTED_DRIVES * AVG_DRIVE_SIZE_MB))

clear
echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}KITTI DOWNLOAD PROGRESS MONITOR${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""

# Function to get human readable size
human_size() {
    local size_bytes=$1
    if [ $size_bytes -lt 1024 ]; then
        echo "${size_bytes}B"
    elif [ $size_bytes -lt 1048576 ]; then
        echo "$(awk "BEGIN {printf \"%.1f\", $size_bytes/1024}")KB"
    elif [ $size_bytes -lt 1073741824 ]; then
        echo "$(awk "BEGIN {printf \"%.1f\", $size_bytes/1048576}")MB"
    else
        echo "$(awk "BEGIN {printf \"%.2f\", $size_bytes/1073741824}")GB"
    fi
}

# Function to estimate time remaining
estimate_time() {
    local current_mb=$1
    local total_mb=$2
    local elapsed_sec=$3

    if [ $current_mb -eq 0 ]; then
        echo "calculating..."
        return
    fi

    local mb_per_sec=$(awk "BEGIN {printf \"%.2f\", $current_mb/$elapsed_sec}")
    local remaining_mb=$((total_mb - current_mb))
    local remaining_sec=$(awk "BEGIN {printf \"%.0f\", $remaining_mb/$mb_per_sec}")

    # Convert to human readable
    if [ $remaining_sec -lt 60 ]; then
        echo "${remaining_sec}s"
    elif [ $remaining_sec -lt 3600 ]; then
        local mins=$((remaining_sec / 60))
        local secs=$((remaining_sec % 60))
        echo "${mins}m ${secs}s"
    else
        local hours=$((remaining_sec / 3600))
        local mins=$(((remaining_sec % 3600) / 60))
        echo "${hours}h ${mins}m"
    fi
}

START_TIME=$(date +%s)

# Monitor loop
while true; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))

    # Count completed drives
    COMPLETED_DRIVES=$(find "${KITTI_RAW}" -mindepth 2 -maxdepth 2 -type d -name "*_sync" 2>/dev/null | wc -l)

    # Count active downloads (zip files being downloaded)
    ACTIVE_DOWNLOADS=$(find "${KITTI_RAW}" -name "*.zip" 2>/dev/null | wc -l)

    # Get current disk usage
    if [ -d "${KITTI_RAW}" ]; then
        CURRENT_SIZE=$(du -sb "${KITTI_RAW}" 2>/dev/null | cut -f1)
        CURRENT_SIZE_MB=$((CURRENT_SIZE / 1048576))
    else
        CURRENT_SIZE=0
        CURRENT_SIZE_MB=0
    fi

    # Calculate progress
    PROGRESS_PCT=$(awk "BEGIN {printf \"%.1f\", ($COMPLETED_DRIVES/$EXPECTED_DRIVES)*100}")

    # Estimate speed and time remaining
    if [ $ELAPSED -gt 5 ] && [ $CURRENT_SIZE_MB -gt 0 ]; then
        SPEED_MBPS=$(awk "BEGIN {printf \"%.2f\", $CURRENT_SIZE_MB/$ELAPSED}")
        TIME_REMAINING=$(estimate_time $CURRENT_SIZE_MB $TOTAL_SIZE_MB $ELAPSED)
    else
        SPEED_MBPS="calculating..."
        TIME_REMAINING="calculating..."
    fi

    # Format elapsed time
    if [ $ELAPSED -lt 60 ]; then
        ELAPSED_STR="${ELAPSED}s"
    elif [ $ELAPSED -lt 3600 ]; then
        MINS=$((ELAPSED / 60))
        SECS=$((ELAPSED % 60))
        ELAPSED_STR="${MINS}m ${SECS}s"
    else
        HOURS=$((ELAPSED / 3600))
        MINS=$(((ELAPSED % 3600) / 60))
        ELAPSED_STR="${HOURS}h ${MINS}m"
    fi

    # Clear screen and display status
    clear
    echo -e "${BLUE}================================================================================================${NC}"
    echo -e "${BLUE}KITTI DOWNLOAD PROGRESS MONITOR${NC}"
    echo -e "${BLUE}================================================================================================${NC}"
    echo ""

    # Progress bar
    BAR_WIDTH=50
    FILLED=$((COMPLETED_DRIVES * BAR_WIDTH / EXPECTED_DRIVES))
    BAR=$(printf "%${FILLED}s" | tr ' ' '█')
    EMPTY=$(printf "%$((BAR_WIDTH - FILLED))s" | tr ' ' '░')

    echo -e "${CYAN}Progress:${NC}"
    echo -e "  [${GREEN}${BAR}${NC}${EMPTY}] ${PROGRESS_PCT}%"
    echo ""

    echo -e "${CYAN}Status:${NC}"
    echo -e "  Completed drives:  ${GREEN}${COMPLETED_DRIVES}${NC}/${EXPECTED_DRIVES}"
    echo -e "  Active downloads:  ${YELLOW}${ACTIVE_DOWNLOADS}${NC} (max 4 concurrent)"
    echo ""

    echo -e "${CYAN}Size:${NC}"
    echo -e "  Downloaded:        $(human_size $CURRENT_SIZE) / ~${TOTAL_SIZE_MB}MB (~$((TOTAL_SIZE_MB/1024))GB)"
    echo ""

    echo -e "${CYAN}Speed:${NC}"
    if [ "$SPEED_MBPS" != "calculating..." ]; then
        echo -e "  Download speed:    ${GREEN}${SPEED_MBPS} MB/s${NC}"
    else
        echo -e "  Download speed:    ${YELLOW}${SPEED_MBPS}${NC}"
    fi
    echo ""

    echo -e "${CYAN}Time:${NC}"
    echo -e "  Elapsed:           ${ELAPSED_STR}"
    if [ "$TIME_REMAINING" != "calculating..." ]; then
        echo -e "  Estimated remaining: ${GREEN}${TIME_REMAINING}${NC}"
    else
        echo -e "  Estimated remaining: ${YELLOW}${TIME_REMAINING}${NC}"
    fi
    echo ""

    # List recently completed drives
    if [ $COMPLETED_DRIVES -gt 0 ]; then
        echo -e "${CYAN}Recently completed:${NC}"
        find "${KITTI_RAW}" -mindepth 2 -maxdepth 2 -type d -name "*_sync" -printf "%T+ %p\n" 2>/dev/null | \
            sort -r | head -5 | while read timestamp path; do
            drive_name=$(basename "$path")
            echo -e "  ${GREEN}✓${NC} ${drive_name}"
        done
        echo ""
    fi

    # Check if downloads are complete
    if [ $COMPLETED_DRIVES -ge $EXPECTED_DRIVES ] && [ $ACTIVE_DOWNLOADS -eq 0 ]; then
        echo -e "${GREEN}================================================================================================${NC}"
        echo -e "${GREEN}DOWNLOAD COMPLETE!${NC}"
        echo -e "${GREEN}================================================================================================${NC}"
        echo ""
        echo -e "Total time: ${ELAPSED_STR}"
        echo -e "Total size: $(human_size $CURRENT_SIZE)"
        echo -e "Average speed: ${SPEED_MBPS} MB/s"
        echo ""
        break
    fi

    # Show tip
    echo -e "${YELLOW}Tip: Press Ctrl+C to exit monitor (downloads will continue in main terminal)${NC}"

    # Update every 2 seconds
    sleep 2
done
