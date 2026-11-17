BEGIN {
    print "/* Clenshaw-Curtis tables: verbatim prefixes of the provided tables. */"
    print "static const int clencurt_M = 12;"
}
/^static const double clencurt_x/ {
    section = 1; count = 0;
    print "static const double clencurt_x[4096] = {";
    next;
}
/^static const double clencurt_w/ {
    section = 2; count = 0;
    print "static const double clencurt_w[8204] = {";
    next;
}
/^};/ {
    if (section) print "};";
    section = 0;
    next;
}
section == 1 { if (count++ < 4096) print; next; }
section == 2 { if (count++ < 8204) print; next; }
