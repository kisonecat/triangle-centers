{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell {
  name = "python-bs4-env";

  packages = [
    (pkgs.python312.withPackages (ps: with ps; [
      beautifulsoup4
      lxml           # parser backend for BeautifulSoup (fast & robust)
      # html5lib     # optional: alternate parser; uncomment if you prefer it
    ]))
  ];

  # (Optional) keep user site-packages from leaking in
  shellHook = ''
    export PYTHONNOUSERSITE=1
  '';
}

