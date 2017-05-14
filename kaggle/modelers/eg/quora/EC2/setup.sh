git clone https://github.com/tomasr/molokai.git
rm -rf $HOME/.vim
mv molokai $HOME/.vim
rm -rf molokai

ROOT_DIR="$(git rev-parse --show-toplevel)"/kaggle/modelers/eg/quora
cp $ROOT_DIR/EC2/.vimrc $HOME

# for f in $(ls -A)
# do
# 	if [ $f != "$(basename "$0")" ]
# 		then
# 			cp -n $ROOT_DIR/EC2/$f $HOME
# 	fi
# done

conda install -y gensim=0
conda install -y numpy
conda update -y scikit-learn
pip install regex

python -c "import nltk; nltk.download('all')"

mkdir -p $ROOT_DIR/Data
cd $ROOT_DIR/Data
wget -nc https://www.dropbox.com/s/cpvtr4qpunpkcac/train.csv
wget -nc https://www.dropbox.com/s/ze8odkdw1444e92/test.csv

mkdir -p $ROOT_DIR/Data/word2vec
cd $ROOT_DIR/Data/word2vec
wget -nc https://www.dropbox.com/s/k9fp6t75fdmbt3q/GoogleNews-vectors-negative300.bin.gz
gunzip -k GoogleNews-vectors-negative300.bin.gz

mkdir -p $ROOT_DIR/Data/glove/gensim
cd $ROOT_DIR/Data/glove/gensim
wget -nc http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
echo '400000 300' | cat - glove.6B.300d.txt > temp && mv temp glove.6B.300d.txt

cd $ROOT_DIR/EC2
