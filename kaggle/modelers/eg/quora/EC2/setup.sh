git clone https://github.com/tomasr/molokai.git
rm -rf $HOME/.vim
mv molokai $HOME/.vim
rm -rf molokai

cp -r ./home/* $HOME

ROOT_DIR="$(git rev-parse --show-toplevel)"/kaggle/modelers/eg/quora

wget https://www.dropbox.com/s/cpvtr4qpunpkcac/train.csv -O $ROOT_DIR/Data/train.csv
wget https://www.dropbox.com/s/ze8odkdw1444e92/test.csv -O $ROOT_DIR/Data/test.csv
