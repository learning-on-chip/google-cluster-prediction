job_events.sqlite3: data/output/job_events.sqlite3 job_events.sql
	cp $< $@
	cat job_events.sql | sqlite3 $@

data/*:
	${MAKE} -C data $*
