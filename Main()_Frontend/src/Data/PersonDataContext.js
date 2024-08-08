import React, { createContext, useState, useEffect } from 'react';

const PersonDataContext = createContext();

const PersonDataProvider = ({ children }) => {
  const [personData, setPersonData] = useState(() => {
    const savedPersonData = localStorage.getItem('personData');
    if (savedPersonData) {
      try {
        return JSON.parse(savedPersonData);
      } catch (error) {
        console.error('Failed to parse personData from localStorage:', error);
        return [];
      }
    }
    return [];
  });

  useEffect(() => {
    console.log('PersonDataContext에 저장된 데이터:', personData);
  }, [personData]);

  useEffect(() => {
    try {
      localStorage.setItem('personData', JSON.stringify(personData));
    } catch (error) {
      console.error('Failed to save personData to localStorage:', error);
    }
  }, [personData]);

  const updatePersonData = (newData) => {
    setPersonData(newData);
  };

  const clearPersonData = () => {
    setPersonData([]);
  };

  return (
    <PersonDataContext.Provider
      value={{
        personData,
        setPersonData: updatePersonData,
        clearPersonData,
      }}
    >
      {children}
    </PersonDataContext.Provider>
  );
};

export { PersonDataContext, PersonDataProvider };
