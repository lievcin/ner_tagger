gmb:
	wget -O ./data/gmb-1.0.0.zip "https://gmb.let.rug.nl/releases/gmb-1.0.0.zip"
	unzip ./data/gmb-1.0.0.zip -d ./data/
	rm ./data/gmb-1.0.0.zip
	python transformations/gmb.py

glove:
	wget -O ./pre-trained-embeddings/glove.6B.zip "http://nlp.stanford.edu/data/glove.6B.zip"
	unzip ./pre-trained-embeddings/glove.6B.zip -d ./pre-trained-embeddings/glove.6B/
	rm ./pre-trained-embeddings/glove.6B.zip

elmo:
	wget -O ./pre-trained-embeddings/elmo_3.tar.gz "https://tfhub.dev/google/elmo/3?tf-hub-format=compressed"
	mkdir -p ./pre-trained-embeddings/elmo_3
	tar -xzf ./pre-trained-embeddings/elmo_3.tar.gz -C pre-trained-embeddings/elmo_3/
	rm ./pre-trained-embeddings/elmo_3.tar.gz