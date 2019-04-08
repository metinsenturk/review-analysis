say_hello: 
	@echo "Hello World"

who_is_your_daddy:
	@echo "Metin Senturk"

install_mallet:
	sudo apt-get install default-jdk
	sudo apt-get install ant
	git clone https://github.com/mimno/Mallet.git model/mallet
	cd model/mallet
	ant

install_spacy_requirements:
	python3 -m spacy download en_core_web_sm
	python3 -m spacy download en_core_web_lg