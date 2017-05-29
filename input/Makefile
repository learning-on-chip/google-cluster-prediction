data:
	${MAKE} -C source download TABLE=task_usage
	${MAKE} -C source distribute TABLE=task_usage
	ln -sf source/output/task_usage/distribution $@

extra.sqlite3:
	${MAKE} -C source convert TABLE=job_events
	ln -sf source/output/job_events.sqlite3 $@
