// Navbar.js

import React, { useState, useContext } from "react";
import "./Navbar.css";
import { Link, useNavigate } from "react-router-dom";
import { DataContext } from "../Data/DataContext";

function Navbar({
  resetSelection,
  onUploadClick,
  onMyPageClick,
  realtimeSelection,
}) {
  const navigate = useNavigate();
  const { clearData, data } = useContext(DataContext);

  const handleLogout = async (event) => {
    event.preventDefault();
    localStorage.clear();
    sessionStorage.clear();
    clearData();

    // 캐시 지우기
    if ("caches" in window) {
      const cacheNames = await caches.keys();
      await Promise.all(cacheNames.map((name) => caches.delete(name)));
    }

    navigate("/");
  };

  const handleRealClick = (event) => {
    event.preventDefault();
    realtimeSelection();
  };

  const handleHomeClick = (event) => {
    event.preventDefault();
    resetSelection(); // 상태 초기화
    navigate("/home");
  };

  return (
    <div className="Navbar">
      <div className="NavbarLogo">
        <Link to="/home" onClick={handleHomeClick}>
          메인페이지
        </Link>
      </div>
      <div className="NavbarUser">
        {data.userName ? `${data.userName}` : "Guest님"}
      </div>
      <ul className="NavbarMenuList">
        <li>
          <Link to="#" onClick={handleRealClick}>
            실시간 분석
          </Link>
        </li>
      </ul>
      <ul className="NavbarMenuList">
        <li>
          <Link to="#" onClick={onUploadClick}>
            영상 분석
          </Link>
        </li>
      </ul>
      <ul className="NavbarMenuList">
        <li>
          <Link to="#" onClick={onMyPageClick}>
            마이페이지
          </Link>
        </li>
      </ul>
      <ul className="NavbarMenuList">
        <li>
          <Link to="#" onClick={handleLogout}>
            로그아웃
          </Link>
        </li>
      </ul>
    </div>
  );
}

export default Navbar;
