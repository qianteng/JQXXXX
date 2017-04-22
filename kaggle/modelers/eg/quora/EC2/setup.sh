git clone https://github.com/tomasr/molokai.git
rm -rf $HOME/.vim
mv molokai $HOME/.vim
rm -rf molokai

ROOT_DIR="$(git rev-parse --show-toplevel)"/kaggle/modelers/eg/quora

for f in .bash_profile .gitconfig .vimrc git-completion.bash git-prompt.sh
do
	cp -n $ROOT_DIR/EC2/$f $HOME
done

wget https://www.dropbox.com/s/cpvtr4qpunpkcac/train.csv -O $ROOT_DIR/Data/train.csv
wget https://www.dropbox.com/s/ze8odkdw1444e92/test.csv -O $ROOT_DIR/Data/test.csv
