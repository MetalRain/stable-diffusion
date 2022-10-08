#!/bin/bash
# Create explore folder and scripts for single prompt
prompt_md5=`/bin/echo $1 | /usr/bin/md5sum | /bin/cut -f1 -d" "`
base_folder="~/repos/stable-diffusion"
mkdir -p $base_folder/explore/$prompt_md5
echo "$1" > $base_folder/explore/$prompt_md5/prompt.txt
printf "#!/bin/bash\ncd $base_folder\nbash $base_folder/scripts/explore-portraits.sh '$1' \$1" > $base_folder/explore/$prompt_md5/explore.sh
printf "#!/bin/bash\ncd $base_folder\nbash $base_folder/scripts/vary.sh \$1 \"$(cat $base_folder/explore/$prompt_md5/prompt.txt)\$2\" \$3" > $base_folder/explore/$prompt_md5/vary.sh
chmod +x $base_folder/explore/$prompt_md5/explore.sh
chmod +x $base_folder/explore/$prompt_md5/vary.sh
echo $prompt_md5