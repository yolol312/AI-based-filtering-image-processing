import React, { useContext, useEffect, useCallback } from "react";
import "./FileModal.css";
import FileUploadFiltering from "./FileUploadFiltering";
import { DataContext } from "../Data/DataContext";

const Server_IP = process.env.REACT_APP_Server_IP;

function FileModal({ isOpen, onClose }) {
  const { data, setData } = useContext(DataContext);
  const userId = data.userId;

  const fetchProPersonInfo = useCallback(async () => {
    if (!userId) {
      console.error("User ID is not available");
      return;
    }

    try {
      const response = await fetch(
        `${Server_IP}/update_pro_video?user_id=${userId}`,
        {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const data1 = await response.json();
      console.log("받아온 provideopersoninfo:", data1.map_camera_provideo_info);
      setData({ mapcameraprovideoinfo: data1.map_camera_provideo_info });
    } catch (error) {
      console.error("Error:", error);
    }
  }, [userId, setData]);

  const handleClose = useCallback(() => {
    fetchProPersonInfo();
    onClose();
  }, [fetchProPersonInfo, onClose]);

  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key === "Escape") {
        handleClose();
      }
    };

    if (isOpen) {
      window.addEventListener("keydown", handleKeyDown);
    }

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [isOpen, handleClose]);

  return (
    <div
      className="File-modal-overlay"
      style={{
        visibility: isOpen ? "visible" : "hidden",
        opacity: isOpen ? 1 : 0,
        transition: "opacity 0.3s ease, visibility 0.3s ease",
      }}
    >
      <div className="File-modal-content">
        <div className="File-modal-top">
          <button className="File-modal-button-close" onClick={handleClose}>
            X
          </button>
        </div>
        <div className="File-modal-body">
          <FileUploadFiltering handleClose={handleClose} />
        </div>
        <div className="File-modal-bottom"></div>
      </div>
    </div>
  );
}

export default FileModal;
