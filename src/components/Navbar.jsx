import React from 'react';
import { Link } from 'react-router-dom';

const Navbar = () => {
  return (
    <nav className="navbar">
      <div className="nav-container">
        <div className="nav-logo">
          <h1>SignLink</h1>
        </div>
        <div className="nav-menu">
          <Link to="/" className="nav-link">Home</Link>
          <Link to="/download" className="nav-link">Download</Link>
          <a href="#" className="nav-link">FAQ</a>
          <a href="#" className="nav-link">Contact</a>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;