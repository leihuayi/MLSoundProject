pdf: project_final.tex project_final.bib
	pdflatex project_final
	bibtex project_final
	pdflatex project_final
	pdflatex project_final
	@echo "================================================"
	@echo "All done! project_final.pdf created"
	@echo "================================================"

clean:
	rm -rf project_final.log project_final.pdf project_final.blg project_final.bbl project_final.aux project_final.out
