---
name: verify-tests
description: Work out whether a passing test suite is evidence the code works or evidence it was made to pass. Use when a handover reports tests passing, when a test that used to fail now does not, or when new tests arrived with a change.
---

# Verifying a pass

A worker who ran the tests and reported what happened has done part of the job. A green
suite is a different claim from "the code works": it also goes green when a failing test
was bent to fit the code, when the code was bent to fit a test, or when the new tests
never had teeth. One section per case follows.

## Was a failing test changed to fit the code

A test stops objecting when an assertion is removed, a bound loosened, a case skipped or
marked expected-failure, an expected value edited to whatever the code now produces.

Whether any of that is defensible turns on what the test was written to prevent, and that
lives in the commit that introduced it -- so the history settles this, not the diff. A
test edited while its reason still stands is the thing to raise. "The test was wrong" is
sometimes true and is worth hearing said rather than assumed.

## Was the code changed only enough to pass one

This one hides, because the test is untouched and still green. The check is satisfied
while the thing it checks is not: only the input the test uses gets handled, the expected
value moves into the code, the failing path is caught or skipped rather than fixed.

A neighbouring input from the same class separates the two -- real code handles it, code
aimed at the test does not. Where reaching that case would mean writing something, asking
for it is cheaper.

## Do the new tests earn their place

Three angles decide it, and each has its own answer.

**Relevance.** Whether the case actually asked about is in one of them. If it is not, that
case is worth naming and requesting by name; a suite that covers the general shape and not
that case has not been aimed at the request.

**Teeth.** Whether any of them could fail. A test that asserts what it just set up, or
mocks away the thing under test, passes whatever the code does -- and is worth saying so
about, because it reports success either way.

**Necessity.** Whether they needed to exist. Tests nobody asked for pad a handover and
raise the cost of every later change, so what each one is for is a fair question.


