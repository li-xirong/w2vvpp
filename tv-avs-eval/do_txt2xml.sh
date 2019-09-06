rootpath=$HOME/VisualSearch

etime=1.0

input=$1
edition=$2
overwrite=0
if [ $# -gt 2 ]; then
    overwrite=$3
fi
python txt2xml.py $input --edition $edition --priority 1 --etime $etime --desc "This run uses the top secret x-component" --rootpath $rootpath --overwrite $overwrite


