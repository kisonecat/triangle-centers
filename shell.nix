{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell {
  name = "python-bs4-env";

  packages = [
    pkgs.curl
    pkgs.cacert

    (pkgs.python312.withPackages (ps: with ps; [
      beautifulsoup4
      lxml           # parser backend for BeautifulSoup (fast & robust)
      # html5lib     # optional: alternate parser; uncomment if you prefer it
    ]))
  ];

  # Make TLS certs visible inside the shell
  SSL_CERT_FILE     = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
  NIX_SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";

  # (Optional) keep user site-packages from leaking in
  shellHook = ''
    export PYTHONNOUSERSITE=1
    export CURL_CA_BUNDLE="$SSL_CERT_FILE"
    export GIT_SSL_CAINFO="$SSL_CERT_FILE"
  '';
}

