.PHONY: theme mvrk site
default: site

theme:
	git pull origin main
	git submodule update --init --recursive
	cd Galileo && git pull origin latest --rebase
	git add .
	git commit -m "Update theme"
	git push -u origin main

mvrk:
	git pull origin main
	git submodule update --init --recursive
	cd Maverick && git pull origin master --rebase
	git add .
	git commit -m "Update Maverick"
	git push -u origin main

site:
	git pull origin main
	git add .
	git commit -m "Update site ${msg}"
	git push -u origin main
