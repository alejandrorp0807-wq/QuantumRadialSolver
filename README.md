# QuantumRadialSolver

Numerical solution of the **radial Schrödinger equation** for hydrogen, hydrogen-like ions, and lithium using the **Numerov method** combined with a **bisection root-finding scheme**.

This project computes bound-state eigenvalues (`alpha` parameters), compares them to analytical solutions when available, and plots the **radial wavefunctions** for the first few quantum states.

---

## 🔹 Features
- Implements **Numerov integration** for solving the radial Schrödinger equation.  
- Uses **bisection** to locate eigenvalues with controlled accuracy.  
- Models:
  - Hydrogen atom (`H`)
  - Hydrogen-like atoms (effective potential correction)
  - Lithium atom approximation (screened Coulomb potential)
- Supports different angular momentum quantum numbers `l`.  
- Computes and compares **numerical vs analytical energies**.  
- Plots **radial wavefunctions** for multiple bound states.  

---

## 🔹 Requirements
- Python 3.8+  
- NumPy  
- Matplotlib  

Install with:
```bash
pip install numpy matplotlib
