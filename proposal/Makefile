pdf: project_proposal.tex project_proposal.bib
	pdflatex project_proposal
	bibtex project_proposal
	pdflatex project_proposal
	pdflatex project_proposal
	@echo "================================================"
	@echo "All done! project_proposal.pdf created"
	@echo "================================================"

clean:
	rm -rf project_proposal.log project_proposal.pdf project_proposal.blg project_proposal.bbl project_proposal.aux project_proposal.out
