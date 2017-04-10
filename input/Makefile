data:
	${MAKE} -C source download distribute TABLE=task_usage
	ln -sf source/output/task_usage/distribution $@

meta.sqlite3:
	${MAKE} -C source convert TABLE=job_events
	ln -sf source/output/job_events.sqlite3 $@
