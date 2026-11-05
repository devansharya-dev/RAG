import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "data" / "evaluation_documents"
DATASET_PATH = ROOT / "tests" / "evaluation" / "datasets" / "rag_eval.json"

DOCS = {
    "project_nebula.txt": """
PROJECT NEBULA — INFRASTRUCTURE PROFILE

Project Nebula is a cloud-based analytics platform designed to process
large volumes of business events. Its production infrastructure uses
48 processing nodes distributed across three availability zones.
PostgreSQL is the primary relational database used for transactional
records and account information. The platform was officially launched
in March 2026. Redis is used for temporary session storage, while
Apache Kafka handles asynchronous event streaming. The engineering
team uses Python for data processing and FastAPI for internal services.
""",

    "project_nebula_architecture.txt": """
PROJECT NEBULA — ARCHITECTURE DETAILS

Project Nebula separates ingestion, processing, and reporting workloads.
Its ingestion layer receives events through HTTP endpoints and Kafka.
The processing layer runs distributed workers across multiple nodes.
The production deployment uses 48 processing nodes, but the staging
environment uses only 8 nodes. PostgreSQL stores transactional data.
The platform uses Redis for short-lived session information.
Its monitoring stack includes Prometheus and Grafana.
""",

    "project_orion.txt": """
PROJECT ORION — SERVICE PROFILE

Project Orion is an inventory management platform for retail businesses.
Its backend is divided into 12 independent microservices. The backend
framework used by the project is FastAPI. Redis provides caching for
frequently requested inventory information. PostgreSQL stores product
and order records, while Elasticsearch supports product search.
The application is deployed using Docker containers and Kubernetes.
The project team follows a weekly release cycle.
""",

    "project_orion_infrastructure.txt": """
PROJECT ORION — INFRASTRUCTURE

Project Orion runs its services inside Kubernetes clusters.
The platform contains 12 microservices, including inventory, orders,
payments, authentication, and reporting services. Redis is configured
as the primary caching technology. The team uses RabbitMQ for selected
background jobs. PostgreSQL is used for persistent relational storage.
The infrastructure team monitors container health through Grafana.
""",

    "project_atlas.txt": """
PROJECT ATLAS — DATA PLATFORM

Project Atlas is a data warehouse platform for financial reporting.
Its main analytical database is ClickHouse. Daily ingestion jobs are
scheduled through Apache Airflow. The platform processes approximately
2 terabytes of raw event data each day. Its reporting dashboard is
built with React. Data validation is performed before warehouse loading.
The platform uses object storage for historical raw files.
""",

    "project_atlas_security.txt": """
PROJECT ATLAS — SECURITY PROFILE

Project Atlas protects financial reporting data through role-based
access control. Administrative actions are recorded in an audit log.
Data transferred between services is encrypted using TLS.
The platform's analytical database is ClickHouse. Access credentials
are stored in a centralized secrets manager. Security reviews are
conducted before major production releases.
""",

    "project_phoenix.txt": """
PROJECT PHOENIX — CUSTOMER SUPPORT

Project Phoenix is a customer support automation platform.
Its natural language processing service is built using Python.
The application uses MongoDB to store conversation histories.
The team selected RabbitMQ for asynchronous task distribution.
Customer support agents access the system through a Vue.js dashboard.
The first production release was completed in January 2025.
""",

    "project_phoenix_ai.txt": """
PROJECT PHOENIX — AI MODULE

The AI module of Project Phoenix classifies incoming support requests.
It uses a transformer-based text classification model.
The model assigns requests to categories such as billing,
technical support, and account access. Python is used for model
inference and preprocessing. Conversation history is stored in
MongoDB. RabbitMQ distributes background classification tasks.
""",

    "project_cirrus.txt": """
PROJECT CIRRUS — CLOUD STORAGE

Project Cirrus provides managed object storage for internal applications.
Its metadata service uses PostgreSQL. The storage layer is compatible
with the S3 API. The project uses Go for its storage gateway.
Automated backups run every 12 hours. The service is deployed across
two geographic regions. Its operational dashboard is built with React.
""",

    "project_cirrus_network.txt": """
PROJECT CIRRUS — NETWORK DESIGN

Project Cirrus uses private networking between its storage components.
External API traffic is routed through a reverse proxy.
The storage gateway is implemented in Go. Metadata records are stored
in PostgreSQL. Backups are replicated to a second geographic region.
The service uses TLS for encrypted communication.
""",

    "project_aurora.txt": """
PROJECT AURORA — LEARNING PLATFORM

Project Aurora is an online learning management system.
Its frontend is developed with Next.js and TypeScript.
The backend uses Django REST Framework.
MySQL is the primary database for course and enrollment records.
The platform uses Celery for asynchronous email notifications.
Learning videos are stored in cloud object storage.
""",

    "project_aurora_analytics.txt": """
PROJECT AURORA — ANALYTICS

Project Aurora tracks course completion and learner engagement.
Its analytics jobs are executed using Apache Airflow.
MySQL remains the primary transactional database.
The analytics dashboard displays enrollment trends,
completion percentages, and course activity.
The team exports aggregated reports every Monday.
""",

    "project_vortex.txt": """
PROJECT VORTEX — CYBERSECURITY

Project Vortex is a security monitoring platform.
It collects authentication and network events from enterprise systems.
The event search engine is Elasticsearch.
Log processing pipelines are written in Python.
The platform sends critical alerts through email and Slack.
Its detection rules are maintained by the security engineering team.
""",

    "project_vortex_detection.txt": """
PROJECT VORTEX — DETECTION ENGINE

The Project Vortex detection engine identifies suspicious login patterns.
It evaluates failed login frequency, unusual locations, and device changes.
Elasticsearch indexes the collected security events.
Python workers evaluate detection rules.
The platform retains searchable security logs for 90 days.
Critical findings are forwarded to the incident response team.
""",

    "project_meridian.txt": """
PROJECT MERIDIAN — PAYMENT SYSTEM

Project Meridian is a digital payment processing system.
Its core transaction service is written in Java.
The primary relational database is Oracle Database.
Apache Kafka distributes transaction events between services.
The system uses a dedicated reconciliation process every night.
Payment records are retained according to internal compliance policies.
""",

    "project_meridian_compliance.txt": """
PROJECT MERIDIAN — COMPLIANCE

Project Meridian maintains an audit trail for payment operations.
Transaction events are distributed using Apache Kafka.
The reconciliation process compares payment records with settlement data.
Access to sensitive financial records is restricted.
The core transaction service is implemented in Java.
Compliance reports are generated at the end of each month.
""",

    "project_summit.txt": """
PROJECT SUMMIT — HEALTHCARE SCHEDULING

Project Summit is a healthcare appointment scheduling application.
Its backend is built with Node.js and Express.
The primary database is PostgreSQL.
The application uses Redis to cache doctor availability.
Patients can search for available appointment slots.
The notification service sends appointment reminders by email.
""",

    "project_summit_operations.txt": """
PROJECT SUMMIT — OPERATIONS

Project Summit handles appointment booking and cancellation workflows.
Its Node.js services communicate through internal REST APIs.
PostgreSQL stores appointment and patient scheduling records.
Redis caches frequently requested doctor availability.
The operations team reviews service metrics every day.
The system uses automated backups for database recovery.
""",

    "project_titan.txt": """
PROJECT TITAN — LOGISTICS

Project Titan is a logistics tracking platform.
Its mobile application is built using Flutter.
The backend services are written in Kotlin.
The system uses Cassandra for high-volume tracking events.
Location updates are processed through Apache Kafka.
The platform provides shipment status and delivery estimates.
""",

    "project_titan_tracking.txt": """
PROJECT TITAN — TRACKING ENGINE

The tracking engine processes location updates from delivery vehicles.
Apache Kafka transports location events to processing services.
Cassandra stores high-volume tracking records.
Kotlin services calculate shipment progress.
The Flutter application displays shipment locations and status.
Historical tracking records are used for delivery analysis.
""",

    "project_delta.txt": """
PROJECT DELTA — DEVELOPER PLATFORM

Project Delta provides tools for software development teams.
Its source code repository service is written in Go.
The platform uses PostgreSQL for project metadata.
Build jobs execute inside isolated Docker containers.
The continuous integration pipeline is managed by Jenkins.
Developers can review build logs through a web dashboard.
""",

    "project_delta_ci.txt": """
PROJECT DELTA — CONTINUOUS INTEGRATION

Project Delta executes automated builds for software repositories.
Jenkins coordinates the continuous integration pipeline.
Docker containers provide isolated build environments.
Go powers the repository service.
PostgreSQL stores project and build metadata.
Failed builds generate notifications for repository owners.
""",

    "project_horizon.txt": """
PROJECT HORIZON — RECOMMENDATION ENGINE

Project Horizon recommends articles to users based on reading history.
Its recommendation API is developed using Python and FastAPI.
The system stores user activity in MongoDB.
The ranking pipeline uses collaborative filtering.
Redis caches frequently requested recommendation results.
The product team evaluates recommendation quality every month.
""",

    "project_horizon_models.txt": """
PROJECT HORIZON — MODEL PIPELINE

The Project Horizon ranking pipeline combines user activity
with article interaction history. Collaborative filtering is
used to generate candidate recommendations.
Python handles model training and inference.
MongoDB stores user interaction events.
Redis caches the final recommendation results.
The API is implemented with FastAPI.
""",

    "project_ember.txt": """
PROJECT EMBER — IOT MONITORING

Project Ember monitors industrial equipment using IoT sensors.
Its device communication protocol is MQTT.
The ingestion service is written in Rust.
TimescaleDB stores time-series sensor measurements.
The dashboard displays temperature, pressure, and vibration.
Alerts are triggered when configured thresholds are exceeded.
""",

    "project_ember_sensors.txt": """
PROJECT EMBER — SENSOR DATA

Project Ember collects temperature, pressure, and vibration readings.
MQTT transports sensor messages to the ingestion service.
Rust processes incoming device messages.
TimescaleDB stores historical time-series measurements.
The dashboard allows operators to inspect equipment health.
Threshold-based alerts identify abnormal sensor readings.
""",

    "project_lumen.txt": """
PROJECT LUMEN — CONTENT MANAGEMENT

Project Lumen is a content management system for digital publishers.
Its backend is developed with PHP and Laravel.
MariaDB stores articles, user accounts, and publishing metadata.
The frontend uses Vue.js.
The platform supports editorial approval workflows.
Published content is delivered through a CDN.
""",
}

QUESTIONS = [
    {
        "id": "Q001",
        "question": "How many production processing nodes does Project Nebula use?",
        "expected_answer": "48",
        "relevant_filenames": ["project_nebula.txt"],
        "answerable": True,
    },
    {
        "id": "Q002",
        "question": "What is the primary relational database of Project Nebula?",
        "expected_answer": "PostgreSQL",
        "relevant_filenames": ["project_nebula.txt"],
        "answerable": True,
    },
    {
        "id": "Q003",
        "question": "When was Project Nebula officially launched?",
        "expected_answer": "March 2026",
        "relevant_filenames": ["project_nebula.txt"],
        "answerable": True,
    },
    {
        "id": "Q004",
        "question": "How many microservices does Project Orion contain?",
        "expected_answer": "12",
        "relevant_filenames": ["project_orion.txt"],
        "answerable": True,
    },
    {
        "id": "Q005",
        "question": "Which backend framework does Project Orion use?",
        "expected_answer": "FastAPI",
        "relevant_filenames": ["project_orion.txt"],
        "answerable": True,
    },
    {
        "id": "Q006",
        "question": "Which technology provides caching in Project Orion?",
        "expected_answer": "Redis",
        "relevant_filenames": ["project_orion.txt"],
        "answerable": True,
    },
    {
        "id": "Q007",
        "question": "Which database powers Project Atlas analytics?",
        "expected_answer": "ClickHouse",
        "relevant_filenames": ["project_atlas.txt"],
        "answerable": True,
    },
    {
        "id": "Q008",
        "question": "Which workflow scheduler does Project Atlas use?",
        "expected_answer": "Apache Airflow",
        "relevant_filenames": ["project_atlas.txt"],
        "answerable": True,
    },
    {
        "id": "Q009",
        "question": "Which database stores Project Phoenix conversation histories?",
        "expected_answer": "MongoDB",
        "relevant_filenames": ["project_phoenix.txt"],
        "answerable": True,
    },
    {
        "id": "Q010",
        "question": "Which message broker does Project Phoenix use?",
        "expected_answer": "RabbitMQ",
        "relevant_filenames": ["project_phoenix.txt"],
        "answerable": True,
    },
    {
        "id": "Q011",
        "question": "Which programming language powers the Project Cirrus storage gateway?",
        "expected_answer": "Go",
        "relevant_filenames": ["project_cirrus.txt"],
        "answerable": True,
    },
    {
        "id": "Q012",
        "question": "Which backend framework does Project Aurora use?",
        "expected_answer": "Django REST Framework",
        "relevant_filenames": ["project_aurora.txt"],
        "answerable": True,
    },
    {
        "id": "Q013",
        "question": "Which search engine indexes Project Vortex security events?",
        "expected_answer": "Elasticsearch",
        "relevant_filenames": ["project_vortex.txt"],
        "answerable": True,
    },
    {
        "id": "Q014",
        "question": "Which programming language is used for Project Meridian's transaction service?",
        "expected_answer": "Java",
        "relevant_filenames": ["project_meridian.txt"],
        "answerable": True,
    },
    {
        "id": "Q015",
        "question": "Which database does Project Summit use?",
        "expected_answer": "PostgreSQL",
        "relevant_filenames": ["project_summit.txt"],
        "answerable": True,
    },
    {
        "id": "Q016",
        "question": "Which mobile framework is used by Project Titan?",
        "expected_answer": "Flutter",
        "relevant_filenames": ["project_titan.txt"],
        "answerable": True,
    },
    {
        "id": "Q017",
        "question": "Which tool manages Project Delta's continuous integration pipeline?",
        "expected_answer": "Jenkins",
        "relevant_filenames": ["project_delta.txt"],
        "answerable": True,
    },
    {
        "id": "Q018",
        "question": "Which recommendation algorithm approach does Project Horizon use?",
        "expected_answer": "Collaborative filtering",
        "relevant_filenames": ["project_horizon.txt"],
        "answerable": True,
    },
    {
        "id": "Q019",
        "question": "Which protocol transports Project Ember sensor messages?",
        "expected_answer": "MQTT",
        "relevant_filenames": ["project_ember.txt"],
        "answerable": True,
    },
    {
        "id": "Q020",
        "question": "Which database does Project Lumen use?",
        "expected_answer": "MariaDB",
        "relevant_filenames": ["project_lumen.txt"],
        "answerable": True,
    },
    {
        "id": "Q021",
        "question": "Which cloud provider hosts Project Nebula?",
        "expected_answer": None,
        "relevant_filenames": [],
        "answerable": False,
    },
    {
        "id": "Q022",
        "question": "Who is the CEO of Project Orion?",
        "expected_answer": None,
        "relevant_filenames": [],
        "answerable": False,
    },
    {
        "id": "Q023",
        "question": "What is the annual revenue of Project Atlas?",
        "expected_answer": None,
        "relevant_filenames": [],
        "answerable": False,
    },
    {
        "id": "Q024",
        "question": "How many employees work at Project Phoenix?",
        "expected_answer": None,
        "relevant_filenames": [],
        "answerable": False,
    },
    {
        "id": "Q025",
        "question": "What is the registered office address of Project Titan?",
        "expected_answer": None,
        "relevant_filenames": [],
        "answerable": False,
    },
]


def main():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

    for filename, content in DOCS.items():
        path = DOCS_DIR / filename
        path.write_text(content.strip() + "\n", encoding="utf-8")

    DATASET_PATH.write_text(
        json.dumps(QUESTIONS, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Created documents: {len(DOCS)}")
    print(f"Created questions: {len(QUESTIONS)}")
    print(f"Answerable questions: {sum(q['answerable'] for q in QUESTIONS)}")
    print(f"Unanswerable questions: {sum(not q['answerable'] for q in QUESTIONS)}")
    print(f"Documents directory: {DOCS_DIR}")
    print(f"Evaluation dataset: {DATASET_PATH}")


if __name__ == "__main__":
    main()
