distribution: data/output/task_usage/.downloaded data/output/task_usage/distribution/.processed
	ln -sf data/output/task_usage/distribution $@

data/%:
	${MAKE} -C data $*
