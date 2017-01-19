jobs.sqlite3: data/output/job_events.sqlite3 jobs.sql
	cp $< $@
	cat jobs.sql | sqlite3 $@

data/*:
	${MAKE} -C data $*
