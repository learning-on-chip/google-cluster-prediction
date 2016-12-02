extern crate arguments;
extern crate sql;
extern crate sqlite;
extern crate statistics;

use sqlite::Connection;
use std::path::Path;

type Error = Box<std::error::Error>;

type Result<T> = std::result::Result<T, Error>;

macro_rules! ok(
    ($result:expr) => (
        match $result {
            Ok(result) => result,
            Err(error) => return Err(Box::new(error)),
        }
    );
);

macro_rules! usage(
    () => ({
        usage();
        return Ok(());
    });
);

fn main() {
    if let Err(error) = start() {
        println!("Error: {}", error);
        std::process::exit(1);
    }
}

fn start() -> Result<()> {
    let arguments = ok!(arguments::parse(std::env::args()));

    let data = match arguments.get::<String>("database") {
        Some(path) => {
            if !Path::new(&path).exists() {
                usage!();
            }
            try!(read(path))
        },
        _ => usage!(),
    };

    let (mean, variance) = (statistics::mean(&data), statistics::variance(&data));
    println!("Samples: {}", data.len());
    println!("Average: {:.4} ± {:.4} s", mean, variance.sqrt());

    Ok(())
}

fn read<T: AsRef<Path>>(path: T) -> Result<Vec<f64>> {
    use sql::prelude::*;
    use sqlite::State;

    let connection = ok!(Connection::open(path));
    let sql = ok!(select_from("jobs").columns(&["time"]).order_by(column("time")).compile());
    let mut statement = ok!(connection.prepare(sql));
    let mut data = vec![];
    let mut past = None;
    while let State::Row = ok!(statement.next()) {
        let present = ok!(statement.read::<i64>(0));
        if let Some(past) = past {
            data.push(1e-6 * (present - past) as f64);
        }
        past = Some(present);
    }
    Ok(data)
}

fn usage() {
    println!("Usage: predictor --database <path>");
}
