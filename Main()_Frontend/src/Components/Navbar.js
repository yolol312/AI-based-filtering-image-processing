// Navbar.js

import React, { useState, useContext } from 'react';
import './Navbar.css';
import { Link, useNavigate } from 'react-router-dom';
import MapModal from '../Modal/MapModal';
import FileModal from '../Modal/FileModal';
import { DataContext } from '../Data/DataContext';

function Navbar({ resetSelection, onMyPageClick }) {
  const [isMapModalOpen, setMapModalOpen] = useState(false);
  const [isFileModalOpen, setFileModalOpen] = useState(false);
  const navigate = useNavigate();
  const { clearData, data } = useContext(DataContext);

  const openMapModal = (event) => {
    event.preventDefault();
    setMapModalOpen(true);
  };

  const openFileModal = (event) => {
    event.preventDefault();
    setFileModalOpen(true);
  };

  const closeMapModal = () => setMapModalOpen(false);
  const closeFileModal = () => setFileModalOpen(false);

  const handleLogout = async (event) => {
    event.preventDefault();
    localStorage.clear();
    sessionStorage.clear();
    clearData();

    // 캐시 지우기
    if ('caches' in window) {
      const cacheNames = await caches.keys();
      await Promise.all(cacheNames.map((name) => caches.delete(name)));
    }

    navigate('/');
  };

  const handleHomeClick = (event) => {
    event.preventDefault();
    resetSelection(); // 상태 초기화
    navigate('/home');
  };

  return (
    <div className="Navbar">
      <div className="NavbarLogo">
        <Link to="/home" onClick={handleHomeClick}>
          메인페이지
        </Link>
      </div>
      <ul className="NavbarMenuList">
        <li>
          <Link to="#" onClick={openMapModal}>
            지도업로드
          </Link>
        </li>
        <li>
          <Link to="#" onClick={openFileModal}>
            파일업로드
          </Link>
        </li>
        <li>
          <Link to="#" onClick={onMyPageClick}>
            마이페이지
          </Link>
        </li>
        <li>
          <Link to="#" onClick={handleLogout}>
            로그아웃
          </Link>
        </li>
      </ul>
      <div className="NavbarUser">{data.userName ? `${data.userName}` : 'Guest님'}</div>

      <MapModal isOpen={isMapModalOpen} onClose={closeMapModal} />
      <FileModal isOpen={isFileModalOpen} onClose={closeFileModal} />
    </div>
  );
}

export default Navbar;
