pdf: project_midterm.tex project_midterm.bib
	pdflatex project_midterm
	bibtex project_midterm
	pdflatex project_midterm
	pdflatex project_midterm
	@echo "================================================"
	@echo "All done! project_midterm.pdf created"
	@echo "================================================"

clean:
	rm -rf project_midterm.log project_midterm.pdf project_midterm.blg project_midterm.bbl project_midterm.aux project_midterm.out
