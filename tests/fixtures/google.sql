DELETE FROM job_events WHERE time = 0 OR `event type` <> 0;

CREATE TABLE clone (time INTEGER NOT NULL, app TEXT NOT NULL, user TEXT NOT NULL);
INSERT INTO clone SELECT time, `logical job name`, user from job_events;
DROP TABLE job_events;

CREATE TABLE apps (name TEXT NOT NULL);
INSERT INTO apps SELECT DISTINCT app from clone ORDER BY app;
UPDATE clone SET app = (select rowid from apps where apps.name = clone.app);
DROP TABLE apps;

CREATE TABLE users (name TEXT NOT NULL);
INSERT INTO users SELECT DISTINCT user from clone ORDER BY user;
UPDATE clone SET user = (select rowid from users where users.name = clone.user);
DROP TABLE users;

CREATE TABLE jobs (time INTEGER NOT NULL, app INTEGER NOT NULL, user INTEGER NOT NULL);
INSERT INTO jobs SELECT time, app, user from clone;
DROP TABLE clone;

UPDATE jobs SET app = app - 1, user = user - 1;

VACUUM;
