import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import styles from './HeroSection.module.css';

const HeroSection = () => {
  const {siteConfig} = useDocusaurusContext();

  return (
    <section className={styles.heroSection}>
      <div className="container">
        <div className="row">
          <div className="col col--6">
            <h1 className={styles.heroTitle}>{siteConfig.title}</h1>
            <p className={styles.heroSubtitle}>{siteConfig.tagline}</p>
            <p className={styles.heroDescription}>
              End-to-end design of humanoid robots that perceive, reason, and act in physical environments.
              The book connects AI cognition (LLMs, perception, planning) with robotic embodiment
              (navigation, manipulation, interaction).
            </p>
            <div className={styles.buttons}>
              <Link
                className="button button--primary button--lg"
                to="/docs/introduction">
                Start Learning
              </Link>
              <Link
                className="button button--secondary button--lg"
                to="/docs/module-1-ros2">
                Explore Modules
              </Link>
            </div>
          </div>
          <div className="col col--6">
            <div className={styles.heroImage}>
              <img
                src="/img/hero/robot-illustration.svg"
                alt="Humanoid Robot Illustration"
                width="100%"
                height="auto"
              />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;