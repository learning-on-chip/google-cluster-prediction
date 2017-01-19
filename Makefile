OUTPUT ?= output

run: ${OUTPUT}/jobs.sqlite3
	@cargo run -- --database $<

${OUTPUT}/jobs.sqlite3: data/output/job_events.sqlite3
	mkdir -p ${OUTPUT}
	cp $< $@
	cat assets/jobs.sql | sqlite3 $@

data/*:
	${MAKE} -C data $*

.PHONY: run
