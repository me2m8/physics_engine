use std::{error::Error, fmt::Display};

#[derive(Copy, Clone, Debug)]
pub enum PrimitiveError {
    AppendIndx,
    AppendVert,
    AppendInst,
}

impl Error for PrimitiveError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }

    fn description(&self) -> &str {
        "Error Regarding Primitives"
    }

    fn cause(&self) -> Option<&dyn Error> {
        self.source()
    }
}

impl Display for PrimitiveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                PrimitiveError::AppendIndx => "Couldnt Append Indicies",
                PrimitiveError::AppendVert => "Couldnt Append Verticies",
                PrimitiveError::AppendInst => "Couldnt Append Instances",
            }
        )
    }
}
