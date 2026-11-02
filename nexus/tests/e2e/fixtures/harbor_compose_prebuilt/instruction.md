Use Bash's `/dev/tcp` support to send an HTTP GET request to the Compose sidecar at `api:8000`.

Write only the response body, which should be `hello-from-sidecar`, followed by a newline, to `/app/result.txt`.
Do not install packages and do not modify the sidecar.
