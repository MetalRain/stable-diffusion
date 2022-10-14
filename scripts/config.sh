# Default aspect ratio: portrait, landscape or square
export DEFAULT_ASPECT_RATIO="portrait"

# Default scale
export DEFAULT_SCALE="6"

# Maximum image size your GPUs memory can handle
# Rectangle images: portrait/landscape
export MAX_RECT_DIM="768"
export MIN_RECT_DIM="448"
# Square images
export MAX_SQUARE_DIM="576"

# How many iterations of exploration to do
export MAX_EXPLORE_ITERATIONS="1000"

# Scripts wait to allow GPU to cool off after images
# How many seconds to wait after: 1 image, 3 images, 9 images?
export EXPLORE_WAITS="1,5,10"
export INTERACTIVE_WAITS="0,0,0"
export VARIATION_WAITS="1,3,10"

# How many images will interactive mode create
# before returning to prompt?
export INTERACTIVE_IMAGES="3"

# How many iterations will variations run 
export DEFAULT_VARIATION_ITERATIONS="3"

# Default transformation
export DEFAULT_VARIATION_TRANSFORM="normal"