#!/bin/bash
# Copy Design A Evaluation Subset Data
# Creates a minimal dataset containing only the 35 evaluation samples

set -e

EVAL_LIST="designA_eval_list.txt"
SOURCE_DAT="../data/p2mppdata/test"
SOURCE_IMG="../data/ShapeNetImages/ShapeNetRendering"
SOURCE_IMG_ONLY="${SOURCE_IMG}/rendering_only"
DEST_BASE="../data/designA_subset"
DEST_DAT="${DEST_BASE}/p2mppdata/test"
DEST_IMG="${DEST_BASE}/ShapeNetRendering"
DEST_IMG_ONLY="${DEST_IMG}/rendering_only"

echo "=============================================="
echo "Design A - Copy Evaluation Subset Data"
echo "=============================================="
echo ""
echo "Source data:"
echo "  .dat files:      ${SOURCE_DAT}"
echo "  Images:          ${SOURCE_IMG}"
echo "  Images (extra):  ${SOURCE_IMG_ONLY}"
echo ""
echo "Destination: ${DEST_BASE}"
echo ""

# Check if eval list exists
if [ ! -f "${EVAL_LIST}" ]; then
    echo "❌ Error: ${EVAL_LIST} not found!"
    exit 1
fi

# Create destination directories
echo "Creating destination directories..."
mkdir -p "${DEST_DAT}"
mkdir -p "${DEST_IMG}"
mkdir -p "${DEST_IMG_ONLY}"

# Count total samples
TOTAL=$(wc -l < "${EVAL_LIST}")
echo "📋 Processing ${TOTAL} samples..."
echo ""

# Counters
COUNT=0
COPIED_DAT=0
COPIED_IMG=0
COPIED_IMG_ONLY=0
MISSING_DAT=0
MISSING_IMG=0
MISSING_IMG_ONLY=0

# Process each line in eval list
while IFS= read -r line; do
    COUNT=$((COUNT + 1))
    
    # Skip empty lines
    [ -z "$line" ] && continue
    
    # Extract category and model ID
    # Format: {category}_{model}_{view}.dat
    FILENAME=$(basename "$line")
    CATEGORY=$(echo "$FILENAME" | cut -d'_' -f1)
    MODEL=$(echo "$FILENAME" | cut -d'_' -f2)
    
    echo "[${COUNT}/${TOTAL}] ${CATEGORY}/${MODEL}"
    
    # Copy .dat file
    DAT_SRC="${SOURCE_DAT}/${FILENAME}"
    if [ -f "${DAT_SRC}" ]; then
        cp "${DAT_SRC}" "${DEST_DAT}/"
        COPIED_DAT=$((COPIED_DAT + 1))
        echo "  ✓ .dat copied"
    else
        echo "  ⚠️  .dat not found: ${DAT_SRC}"
        MISSING_DAT=$((MISSING_DAT + 1))
    fi
    
    # Copy image directory
    IMG_SRC="${SOURCE_IMG}/${CATEGORY}/${MODEL}"
    IMG_DEST="${DEST_IMG}/${CATEGORY}"
    
    if [ -d "${IMG_SRC}" ]; then
        mkdir -p "${IMG_DEST}"
        cp -r "${IMG_SRC}" "${IMG_DEST}/"
        COPIED_IMG=$((COPIED_IMG + 1))
        
        # Count images
        IMG_COUNT=$(find "${IMG_DEST}/${MODEL}/rendering" -name "*.png" 2>/dev/null | wc -l)
        echo "  ✓ Images copied (${IMG_COUNT} PNGs)"
    else
        echo "  ⚠️  Images not found: ${IMG_SRC}"
        MISSING_IMG=$((MISSING_IMG + 1))
    fi
    
    # Copy rendering_only images (for safety)
    IMG_ONLY_SRC="${SOURCE_IMG_ONLY}/${CATEGORY}/${MODEL}"
    IMG_ONLY_DEST="${DEST_IMG_ONLY}/${CATEGORY}"
    
    if [ -d "${IMG_ONLY_SRC}" ]; then
        mkdir -p "${IMG_ONLY_DEST}"
        cp -r "${IMG_ONLY_SRC}" "${IMG_ONLY_DEST}/"
        COPIED_IMG_ONLY=$((COPIED_IMG_ONLY + 1))
        
        # Count images
        IMG_ONLY_COUNT=$(find "${IMG_ONLY_DEST}/${MODEL}/rendering" -name "*.png" 2>/dev/null | wc -l)
        echo "  ✓ Extra images copied (${IMG_ONLY_COUNT} PNGs)"
    else
        echo "  ⚠️  Extra images not found: ${IMG_ONLY_SRC}"
        MISSING_IMG_ONLY=$((MISSING_IMG_ONLY + 1))
    fi
    
done < "${EVAL_LIST}"

echo ""
echo "=============================================="
echo "Copy Complete!"
echo "=============================================="
echo ""
echo "📊 Summary:"
echo "  Samples processed:     ${COUNT}"
echo "  .dat files copied:     ${COPIED_DAT}"
echo "  Image dirs copied:     ${COPIED_IMG}"
echo "  Extra image dirs:      ${COPIED_IMG_ONLY}"
echo ""

if [ ${MISSING_DAT} -gt 0 ] || [ ${MISSING_IMG} -gt 0 ] || [ ${MISSING_IMG_ONLY} -gt 0 ]; then
    echo "⚠️  Missing items:"
    [ ${MISSING_DAT} -gt 0 ] && echo "  .dat files:        ${MISSING_DAT}"
    [ ${MISSING_IMG} -gt 0 ] && echo "  Image dirs:        ${MISSING_IMG}"
    [ ${MISSING_IMG_ONLY} -gt 0 ] && echo "  Extra image dirs:  ${MISSING_IMG_ONLY}"
    echo ""
fi

# Calculate size
echo "💾 Storage used:"
DEST_SIZE=$(du -sh "${DEST_BASE}" | cut -f1)
echo "  Total: ${DEST_SIZE}"
echo ""

# Show structure
echo "📁 Directory structure:"
echo "${DEST_BASE}/"
echo "├── p2mppdata/test/              (${COPIED_DAT} .dat files)"
echo "└── ShapeNetRendering/"
for cat in $(ls "${DEST_IMG}" 2>/dev/null | grep -v "rendering_only"); do
    model_count=$(ls "${DEST_IMG}/${cat}" 2>/dev/null | wc -l)
    echo "    ├── ${cat}/                  (${model_count} models)"
done
echo "    └── rendering_only/          (${COPIED_IMG_ONLY} extra image dirs)"

echo ""
echo "✅ Subset data ready at: ${DEST_BASE}"
echo ""
echo "To use this subset, update scripts to point to:"
echo "  data_root: ${DEST_DAT}"
echo "  image_root: ${DEST_IMG}"
echo ""
