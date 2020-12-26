gmb:
	wget -P ./data/ "https://gmb.let.rug.nl/releases/gmb-1.0.0.zip"
	unzip ./data/gmb-1.0.0.zip -d data/
	rm ./data/gmb-1.0.0.zip
	python transformations/gmb.py