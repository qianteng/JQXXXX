git clone https://github.com/tomasr/molokai.git
rm -rf $HOME/.vim
mv molokai $HOME/.vim
rm -rf molokai

cp -r ./home/* $HOME

ROOT_DIR=$HOME/GitHub/JQXXXX/kaggle/modelers/eg/quora

wget --directory-prefix=$ROOT_DIR/Data https://www.dropbox.com/s/cpvtr4qpunpkcac/train.csv
wget --directory-prefix=$ROOT_DIR/Data https://www.dropbox.com/s/ze8odkdw1444e92/test.csv
