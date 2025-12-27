# Makefile for FedXChain IEEE Paper

# Main LaTeX file (without .tex extension)
MAIN = fedxchain_paper

# LaTeX compiler
LATEX = pdflatex
BIBTEX = bibtex

# Default target
all: $(MAIN).pdf

# Compile PDF (with bibliography)
$(MAIN).pdf: $(MAIN).tex references.bib
	@echo "=== First LaTeX pass ==="
	$(LATEX) $(MAIN).tex
	@echo ""
	@echo "=== Running BibTeX ==="
	$(BIBTEX) $(MAIN)
	@echo ""
	@echo "=== Second LaTeX pass ==="
	$(LATEX) $(MAIN).tex
	@echo ""
	@echo "=== Final LaTeX pass ==="
	$(LATEX) $(MAIN).tex
	@echo ""
	@echo "✅ PDF successfully generated: $(MAIN).pdf"
	@echo ""
	@echo "=== File size ==="
	@ls -lh $(MAIN).pdf

# Quick compile (without bibliography, useful for quick iterations)
quick: $(MAIN).tex
	@echo "=== Quick LaTeX compile (no bibliography) ==="
	$(LATEX) $(MAIN).tex
	@echo ""
	@echo "✅ Quick PDF generated: $(MAIN).pdf"

# Clean auxiliary files
clean:
	@echo "=== Cleaning auxiliary files ==="
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).bbl $(MAIN).blg
	rm -f $(MAIN).out $(MAIN).toc $(MAIN).lof $(MAIN).lot
	rm -f $(MAIN).fls $(MAIN).fdb_latexmk $(MAIN).synctex.gz
	@echo "✅ Cleaned"

# Clean everything including PDF
cleanall: clean
	@echo "=== Removing PDF ==="
	rm -f $(MAIN).pdf
	@echo "✅ All files cleaned"

# View PDF (Linux with default viewer)
view: $(MAIN).pdf
	@echo "=== Opening PDF ==="
	xdg-open $(MAIN).pdf &

# Check LaTeX installation
check:
	@echo "=== Checking LaTeX installation ==="
	@which $(LATEX) > /dev/null && echo "✅ pdflatex found: $$(which $(LATEX))" || echo "❌ pdflatex not found"
	@which $(BIBTEX) > /dev/null && echo "✅ bibtex found: $$(which $(BIBTEX))" || echo "❌ bibtex not found"
	@echo ""
	@echo "=== Checking required packages ==="
	@kpsewhich IEEEtran.cls > /dev/null && echo "✅ IEEEtran.cls found" || echo "❌ IEEEtran.cls not found (install texlive-publishers)"
	@kpsewhich algorithm.sty > /dev/null && echo "✅ algorithm.sty found" || echo "❌ algorithm.sty not found (install texlive-science)"
	@kpsewhich algpseudocode.sty > /dev/null && echo "✅ algpseudocode.sty found" || echo "❌ algpseudocode.sty not found (install texlive-science)"
	@kpsewhich amsmath.sty > /dev/null && echo "✅ amsmath.sty found" || echo "❌ amsmath.sty not found (install texlive-latex-base)"
	@kpsewhich booktabs.sty > /dev/null && echo "✅ booktabs.sty found" || echo "❌ booktabs.sty not found (install texlive-latex-recommended)"

# Count words in paper (approximate, excludes LaTeX commands)
wordcount:
	@echo "=== Word count (approximate) ==="
	@detex $(MAIN).tex | wc -w

# Display paper statistics
stats: $(MAIN).pdf
	@echo "=== Paper Statistics ==="
	@echo "PDF size: $$(ls -lh $(MAIN).pdf | awk '{print $$5}')"
	@echo "Pages: $$(pdfinfo $(MAIN).pdf 2>/dev/null | grep Pages | awk '{print $$2}' || echo 'pdfinfo not installed')"
	@echo "Source lines: $$(wc -l < $(MAIN).tex)"
	@echo "References: $$(grep -c '@' references.bib)"
	@echo ""
	@echo "Tables: $$(grep -c '\\begin{table}' $(MAIN).tex)"
	@echo "Figures: $$(grep -c '\\begin{figure}' $(MAIN).tex)"
	@echo "Equations: $$(grep -c '\\begin{equation}' $(MAIN).tex)"
	@echo "Algorithms: $$(grep -c '\\begin{algorithm}' $(MAIN).tex)"

# Help target
help:
	@echo "FedXChain Paper Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make          - Compile full paper with bibliography (default)"
	@echo "  make quick    - Quick compile without bibliography"
	@echo "  make clean    - Remove auxiliary files (keep PDF)"
	@echo "  make cleanall - Remove all generated files including PDF"
	@echo "  make view     - Open PDF with default viewer"
	@echo "  make check    - Check LaTeX installation and packages"
	@echo "  make stats    - Display paper statistics"
	@echo "  make wordcount- Count words in paper"
	@echo "  make help     - Show this help message"
	@echo ""
	@echo "Example workflow:"
	@echo "  1. make check    # Verify LaTeX is installed"
	@echo "  2. make          # Compile full paper"
	@echo "  3. make view     # Open PDF"
	@echo "  4. make stats    # Check paper statistics"

.PHONY: all quick clean cleanall view check wordcount stats help
