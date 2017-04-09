distribution:
	${MAKE} -C data download distribute TABLE=task_usage
	ln -sf data/output/task_usage/distribution $@

meta.sqlite3:
	${MAKE} -C data convert TABLE=job_events
	ln -sf data/output/job_events.sqlite3 $@
