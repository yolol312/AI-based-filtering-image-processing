import React, { useState } from "react";
import "./UploadSwap.css";
import MapModal from "../Modal/MapModal";
import FileModal from "../Modal/FileModal";

function MyPageSwap() {
  const [isMapModalOpen, setMapModalOpen] = useState(false);
  const [isFileModalOpen, setFileModalOpen] = useState(false);
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

  return (
    <div className="UploadSwap">
      <div className="upload-container">
        <div className="upload-box">
          <button className="upload-button" onClick={openMapModal}>
            지도 업로드
          </button>
          <p>
            관리할 지도와 카메라 정보를 입력합니다.
            <br />
            (이미 등록된 지도는 다시 등록하지 않아도 됩니다.)
            <br /> ▶ 바로 파일 업로드
          </p>
        </div>
        <div className="upload-box">
          <button className="upload-button" onClick={openFileModal}>
            영상 업로드
          </button>
          <p>
            분석할 영상을 업로드 합니다.
            <br />
            필터링 정보를 1개 이상 꼭 선택해주시길 바랍니다.
          </p>
        </div>
      </div>
      <MapModal isOpen={isMapModalOpen} onClose={closeMapModal} />
      <FileModal isOpen={isFileModalOpen} onClose={closeFileModal} />
    </div>
  );
}

export default MyPageSwap;
