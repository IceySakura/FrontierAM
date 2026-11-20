#pragma once
#include "optimizer.hpp"

// A reduced solve never calls itself on another transformed model.
Result optimize_core(const Model&,bool check_diagonal=true);
bool reduce_identity(const Model&,Verifier&);
bool reduce_rows(const Model&,Verifier&);
bool substitute_equalities(const Model&,Verifier&);
