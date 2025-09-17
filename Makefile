index.html: header.html footer.html etc.js
	cat header.html etc.js footer.html > index.html

etc.js: etc.json emit-trilinears.py
	python emit-trilinears.py etc.json > etc.js

etc.json: ETC.html find-centers.py
	python find-centers.py ETC.html > etc.json

ETC.html:
	curl -O https://faculty.evansville.edu/ck6/encyclopedia/ETC.html

clean:
	rm etc.js etc.json ETC.html index.html
