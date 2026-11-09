# Sphinx + doxygen documentation build (same layout as ../kaldi-native-fbank).
#
#   make html      -> doxygen (C++ API XML+HTML) + sphinx html; the doxygen
#                     HTML tree is copied next to the sphinx output as
#                     build/html/cpp-api-doxygen and linked from every
#                     cpp-api page.
#   make latexpdf  -> doxygen + sphinx latex pdf.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
DOXYGEN       ?= doxygen

.PHONY: help html latexpdf clean doxygen

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

doxygen:
	@mkdir -p "$(BUILDDIR)/doxygen"
	@$(DOXYGEN) Doxyfile
	# doxygen namespacemembers*.html only link the "ns" label; make the
	# member names themselves clickable (same fixup as kaldi-native-fbank).
	@perl -pi -e 's|<li>([^<]+)&#160;:&nbsp;&#160;<a class="el" href="([^"]+)">|<li><a class="el" href="$$2">$$1</a>&#160;:&nbsp;&#160;<a class="el" href="$$2">|g' \
		"$(BUILDDIR)/doxygen/html"/namespacemembers*.html 2>/dev/null || true

html: doxygen
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	rm -rf "$(BUILDDIR)/html/cpp-api-doxygen"
	cp -r "$(BUILDDIR)/doxygen/html" "$(BUILDDIR)/html/cpp-api-doxygen"

latexpdf: doxygen
	@$(SPHINXBUILD) -M latexpdf "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	rm -rf $(BUILDDIR)

# Catch-all target: route all unknown targets to Sphinx using the "make
# mode" option.
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
