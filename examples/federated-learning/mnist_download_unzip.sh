rm -rf MNIST_DATA
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1yPGvxLqKn2W4_E5jAWMLAuvbYt3yvBOG' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1yPGvxLqKn2W4_E5jAWMLAuvbYt3yvBOG" -O MNIST.zip && rm -rf /tmp/cookies.txt
unzip MNIST.zip
rm -rf MNIST.zip
