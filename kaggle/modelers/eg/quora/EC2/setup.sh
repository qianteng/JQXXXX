git clone https://github.com/tomasr/molokai.git
rm -rf $HOME/.vim
mv molokai $HOME/.vim
rm -rf molokai

ROOT_DIR="$(git rev-parse --show-toplevel)"/kaggle/modelers/eg/quora

for f in $(ls -A)
do
	if [ $f != "$(basename "$0")" ]
		then
			cp $ROOT_DIR/EC2/$f $HOME
	fi
done

CONDA_DIR=$HOME/anaconda2/bin
# $CONDA_DIR/conda install -y numpy

mkdir -p $ROOT_DIR/Data
wget -nc https://www.dropbox.com/s/cpvtr4qpunpkcac/train.csv -O $ROOT_DIR/Data/train.csv
wget -nc https://www.dropbox.com/s/ze8odkdw1444e92/test.csv -O $ROOT_DIR/Data/test.csv
