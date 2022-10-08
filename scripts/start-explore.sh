#!/bin/bash
# Create explore folder and scripts for single prompt
prompt_md5=`/bin/echo $1 | /usr/bin/md5sum | /bin/cut -f1 -d" "`
mkdir -p ~/repos/stable-diffusion/explore/$prompt_md5
echo "$1" > ~/repos/stable-diffusion/explore/$prompt_md5/prompt.txt
chmod +x ~/repos/stable-diffusion/explore/$prompt_md5/prompt.txt
printf "#!/bin/bash\ncd ~/repos/stable-diffusion\nbash ~/repos/stable-diffusion/scripts/explore-portraits.sh '$1' \$1" > ~/repos/stable-diffusion/explore/$prompt_md5/explore.sh
echo $prompt_md5