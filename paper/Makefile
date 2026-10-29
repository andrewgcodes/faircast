# make        -> main.pdf (pdflatex x2: inline thebibliography needs a second pass for \cite)
# make arxiv  -> arxiv.tar.gz, the flat bundle to upload to arXiv (main.tex, main.bbl-free, figures/)
PDFLATEX = pdflatex -interaction=nonstopmode -halt-on-error

main.pdf: main.tex figures/*.png
	$(PDFLATEX) main.tex
	$(PDFLATEX) main.tex

arxiv: main.pdf
	rm -rf arxiv arxiv.tar.gz && mkdir -p arxiv/figures
	cp main.tex arxiv/ && cp figures/fig*.png arxiv/figures/
	cp main.bbl arxiv/ 2>/dev/null || true
	tar czf arxiv.tar.gz -C arxiv .
	@echo "arxiv.tar.gz ready; upload to arXiv with pdflatex as the processor"

clean:
	rm -rf arxiv arxiv.tar.gz main.aux main.log main.out main.pdf main.bbl main.blg

.PHONY: arxiv clean
