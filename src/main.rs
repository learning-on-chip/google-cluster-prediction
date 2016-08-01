extern crate arguments;
extern crate sql;
extern crate sqlite;

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

fn main() {
    if let Err(error) = start() {
        println!("Error: {}", error);
        std::process::exit(1);
    }
}

fn start() -> Result<()> {
    let _ = ok!(arguments::parse(std::env::args()));
    Ok(())
}
