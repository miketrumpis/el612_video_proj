# Simple makefile to quickly access handy build commands for Cython extension
# code generation.  Note that the actual code to produce the extension lives in
# the setup.py file, this Makefile is just meant as a command
# convenience/reminder while doing development.

PKGDIR=video_proj

all: ext

ext:

	python setup.py build_ext --inplace

html:  ${PKGDIR}/mean_shift/cell_labels.html ${PKGDIR}/mean_shift/histogram.html

# Phony targets for cleanup and similar uses

.PHONY: clean

clean:
	- find ${PKGDIR} -name "*.so" -print0 | xargs -0 rm
	- find ${PKGDIR} -name "*.c" -print0 | xargs -0 rm
	- find ${PKGDIR} -name "*.html" -print0 | xargs -0 rm
	- find . -name "*.pyc" -print0 | xargs -0 rm
	rm -rf build

# Suffix rules
%.c : %.pyx
	cython $<

%.html : %.pyx
	cython -a $<

