//! A scripted HTTP/1.1 server on a loopback listener, for the fetcher's
//! tests: it counts every connection it accepts, keeps every request head it
//! reads, and answers each request the way the test's script says. Plain
//! `std::net` threads, so it runs whatever runtime (or none) the test uses.

use std::io::{ErrorKind, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// How the server answers one request.
#[derive(Debug, Clone)]
pub(crate) enum Reply {
    /// `200` with this body and its length.
    Ok(Vec<u8>),
    /// This status with this body and its length.
    Status(u16, Vec<u8>),
    /// This redirect status to this `Location`, with a short body.
    Redirect(u16, String),
    /// A redirect whose head declares a huge body that never comes.
    RedirectWithEndlessBody(u16, String),
    /// A redirect with no `Location` header.
    RedirectWithoutLocation(u16),
    /// `200` declaring `declared` bytes of `Content-Length`, then `body`
    /// (and then the connection held open for a while).
    Declared {
        /// The `Content-Length` value.
        declared: u64,
        /// What is actually sent.
        body: Vec<u8>,
    },
    /// `200` with `Transfer-Encoding: chunked`: `total` bytes in chunks of
    /// `chunk`.
    Chunked {
        /// Bytes in all.
        total: usize,
        /// Bytes per chunk.
        chunk: usize,
    },
    /// `200` with no framing at all (no length, not chunked): `total` bytes,
    /// then the connection is closed.
    Unframed {
        /// Bytes in all.
        total: usize,
    },
    /// `200` declaring `total` bytes, then one byte every `every`.
    Drip {
        /// Bytes declared and (eventually) sent.
        total: usize,
        /// The pause before each byte.
        every: Duration,
    },
    /// Read the request, then send nothing for `hold`.
    Stall(Duration),
    /// Read the request, then send nothing until `gate` opens — returning
    /// early when the client closes the connection or the server stops —
    /// and then `200` with `body`.
    Gated {
        /// Opened by the test.
        gate: Arc<Gate>,
        /// What is sent once it opens.
        body: Vec<u8>,
    },
}

/// A gate a test opens to release every [`Reply::Gated`] connection.
#[derive(Debug, Default)]
pub(crate) struct Gate {
    open: AtomicBool,
}

impl Gate {
    /// Release every connection waiting on this gate (and every later one).
    pub(crate) fn open(&self) {
        self.open.store(true, Ordering::Release);
    }

    fn is_open(&self) -> bool {
        self.open.load(Ordering::Acquire)
    }
}

/// The script: the request head in, the reply out.
type Script = dyn Fn(&str) -> Reply + Send + Sync;

/// A running test server; dropping it stops the accept loop.
pub(crate) struct TestServer {
    addr: SocketAddr,
    accepts: Arc<AtomicUsize>,
    /// Connections whose handling ended (the reply was sent, or the client
    /// closed the connection while the server was dripping or stalling).
    finished: Arc<AtomicUsize>,
    requests: Arc<Mutex<Vec<String>>>,
    stop: Arc<AtomicBool>,
}

impl TestServer {
    /// Serve on a fresh port of `host` (`127.0.0.1` or `[::1]`).
    pub(crate) fn start(
        host: &str,
        script: impl Fn(&str) -> Reply + Send + Sync + 'static,
    ) -> Self {
        let listener = TcpListener::bind(format!("{host}:0")).expect("bind a loopback listener");
        Self::on(listener, script)
    }

    /// Serve on an already-bound listener.
    pub(crate) fn on(
        listener: TcpListener,
        script: impl Fn(&str) -> Reply + Send + Sync + 'static,
    ) -> Self {
        let addr = listener.local_addr().expect("listener address");
        listener
            .set_nonblocking(true)
            .expect("a non-blocking listener");
        let accepts = Arc::new(AtomicUsize::new(0));
        let finished = Arc::new(AtomicUsize::new(0));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let script: Arc<Script> = Arc::new(script);
        {
            let accepts = Arc::clone(&accepts);
            let finished = Arc::clone(&finished);
            let requests = Arc::clone(&requests);
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                while !stop.load(Ordering::Acquire) {
                    match listener.accept() {
                        Ok((stream, _)) => {
                            accepts.fetch_add(1, Ordering::SeqCst);
                            let requests = Arc::clone(&requests);
                            let script = Arc::clone(&script);
                            let stop = Arc::clone(&stop);
                            let finished = Arc::clone(&finished);
                            std::thread::spawn(move || {
                                handle(stream, &requests, script.as_ref(), &stop);
                                finished.fetch_add(1, Ordering::SeqCst);
                            });
                        }
                        Err(e) if e.kind() == ErrorKind::WouldBlock => {
                            std::thread::sleep(Duration::from_millis(2));
                        }
                        Err(_) => std::thread::sleep(Duration::from_millis(2)),
                    }
                }
            });
        }
        Self {
            addr,
            accepts,
            finished,
            requests,
            stop,
        }
    }

    /// The listening port.
    pub(crate) fn port(&self) -> u16 {
        self.addr.port()
    }

    /// `http://<this listener>/<path>` (a bracketed host for IPv6).
    pub(crate) fn url(&self, path: &str) -> String {
        match self.addr {
            SocketAddr::V4(v4) => format!("http://{}:{}{path}", v4.ip(), v4.port()),
            SocketAddr::V6(v6) => format!("http://[{}]:{}{path}", v6.ip(), v6.port()),
        }
    }

    /// Connections accepted so far.
    pub(crate) fn accepts(&self) -> usize {
        self.accepts.load(Ordering::SeqCst)
    }

    /// Wait (at most `limit`) until `n` connections have ended; whether they
    /// did.
    pub(crate) fn wait_finished(&self, n: usize, limit: Duration) -> bool {
        let start = Instant::now();
        while start.elapsed() < limit {
            if self.finished.load(Ordering::SeqCst) >= n {
                return true;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        self.finished.load(Ordering::SeqCst) >= n
    }

    /// The request heads read so far (lower-cased header names kept as sent).
    pub(crate) fn requests(&self) -> Vec<String> {
        self.requests
            .lock()
            .map(|requests| requests.clone())
            .unwrap_or_default()
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
    }
}

/// Read one request head (up to the blank line, at most 64 KiB).
fn read_head(stream: &mut TcpStream) -> String {
    let mut head = Vec::new();
    let mut byte = [0u8; 1];
    while head.len() < 64 * 1024 {
        match stream.read(&mut byte) {
            Ok(1) => {
                head.push(byte[0]);
                if head.ends_with(b"\r\n\r\n") {
                    break;
                }
            }
            _ => break,
        }
    }
    String::from_utf8_lossy(&head).into_owned()
}

/// Sleep for `duration` in small steps, returning early when the server
/// stops.
fn hold(duration: Duration, stop: &AtomicBool) {
    let start = Instant::now();
    while start.elapsed() < duration && !stop.load(Ordering::Acquire) {
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Send nothing for at most `duration`, returning as soon as the client
/// closes the connection (a read sees its end) or the server stops.
fn stall(stream: &mut TcpStream, duration: Duration, stop: &AtomicBool) {
    let _ = stall_until(stream, stop, |elapsed| elapsed >= duration);
}

/// Send nothing until `done` says so (given the time stalled so far),
/// watching the connection: `true` when `done` ended the stall, `false` when
/// the client closed the connection or the server stopped first.
fn stall_until(stream: &mut TcpStream, stop: &AtomicBool, done: impl Fn(Duration) -> bool) -> bool {
    let _ = stream.set_read_timeout(Some(Duration::from_millis(10)));
    let start = Instant::now();
    let mut byte = [0u8; 1];
    loop {
        if done(start.elapsed()) {
            return true;
        }
        if stop.load(Ordering::Acquire) {
            return false;
        }
        match stream.read(&mut byte) {
            Ok(0) => return false,
            Ok(_) => {}
            Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {}
            Err(_) => return false,
        }
    }
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        301 => "Moved Permanently",
        302 => "Found",
        303 => "See Other",
        307 => "Temporary Redirect",
        308 => "Permanent Redirect",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "Status",
    }
}

/// Answer one connection.
fn handle(
    mut stream: TcpStream,
    requests: &Mutex<Vec<String>>,
    script: &Script,
    stop: &AtomicBool,
) {
    let _ = stream.set_nonblocking(false);
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    let head = read_head(&mut stream);
    if let Ok(mut all) = requests.lock() {
        all.push(head.clone());
    }
    let reply = script(&head);
    let mut write = |bytes: &[u8]| stream.write_all(bytes).and_then(|()| stream.flush());
    match reply {
        Reply::Ok(body) => {
            let _ = write(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\n\
                     Connection: close\r\n\r\n",
                    body.len()
                )
                .as_bytes(),
            );
            let _ = write(&body);
        }
        Reply::Status(status, body) => {
            let _ = write(
                format!(
                    "HTTP/1.1 {status} {}\r\nContent-Type: text/plain\r\nContent-Length: {}\r\n\
                     Connection: close\r\n\r\n",
                    reason(status),
                    body.len()
                )
                .as_bytes(),
            );
            let _ = write(&body);
        }
        Reply::Redirect(status, location) => {
            let body = b"moved";
            let _ = write(
                format!(
                    "HTTP/1.1 {status} {}\r\nLocation: {location}\r\nContent-Length: {}\r\n\
                     Connection: close\r\n\r\n",
                    reason(status),
                    body.len()
                )
                .as_bytes(),
            );
            let _ = write(body);
        }
        Reply::RedirectWithEndlessBody(status, location) => {
            let _ = write(
                format!(
                    "HTTP/1.1 {status} {}\r\nLocation: {location}\r\nContent-Length: \
                     100000000\r\n\r\n",
                    reason(status)
                )
                .as_bytes(),
            );
            hold(Duration::from_secs(5), stop);
        }
        Reply::RedirectWithoutLocation(status) => {
            let _ = write(
                format!(
                    "HTTP/1.1 {status} {}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    reason(status)
                )
                .as_bytes(),
            );
        }
        Reply::Declared { declared, body } => {
            let _ =
                write(format!("HTTP/1.1 200 OK\r\nContent-Length: {declared}\r\n\r\n").as_bytes());
            let _ = write(&body);
            hold(Duration::from_secs(5), stop);
        }
        Reply::Chunked { total, chunk } => {
            let _ = write(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n");
            let mut sent = 0;
            while sent < total {
                let n = chunk.min(total - sent);
                let piece = vec![b'x'; n];
                if write(format!("{n:x}\r\n").as_bytes()).is_err()
                    || write(&piece).is_err()
                    || write(b"\r\n").is_err()
                {
                    return;
                }
                sent += n;
            }
            let _ = write(b"0\r\n\r\n");
        }
        Reply::Unframed { total } => {
            let _ = write(b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n");
            let _ = write(&vec![b'y'; total]);
        }
        Reply::Drip { total, every } => {
            let _ = write(format!("HTTP/1.1 200 OK\r\nContent-Length: {total}\r\n\r\n").as_bytes());
            for _ in 0..total {
                hold(every, stop);
                if stop.load(Ordering::Acquire) || write(b"z").is_err() {
                    return;
                }
            }
        }
        Reply::Stall(duration) => stall(&mut stream, duration, stop),
        Reply::Gated { gate, body } => {
            if stall_until(&mut stream, stop, |_| gate.is_open()) {
                let mut write =
                    |bytes: &[u8]| stream.write_all(bytes).and_then(|()| stream.flush());
                let _ = write(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\n\
                         Connection: close\r\n\r\n",
                        body.len()
                    )
                    .as_bytes(),
                );
                let _ = write(&body);
            }
        }
    }
}
